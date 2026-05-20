#!/usr/bin/env python
# -*- coding: utf-8 -*-
# == libmpopt ==
"""
libmpopt fusion wrapper for PPI network alignment.

Replaces the Python `thinqpbo`-based fusion in `ppikktetm_ec_optimizedcopy1.py`
with libmpopt's compiled `qap_dd_fusion --solver qpbo-i` binary.

Source upstream: https://github.com/vislearn/libmpopt/tree/iccv2021
Paper: Hutschenreiter et al. "Fusion Moves for Graph Matching", ICCV 2021.

Public API
----------
- libmpopt_fuse(cfg, current_solution, candidate_solution, ...)
        Drop-in replacement for `asm_fusion_moves_qpbo`. Returns
        a numpy int array of length cfg.n1.

- build_qap_dd_for_fusion(A1, A2, K, lambda_param, p, q)
        Construct a sparse QAP model restricted to {p[i], q[i]}
        per left node. Returns (n_left, n_right, assignments, edges).

- write_dd_file / write_solutions_file
        Low-level I/O helpers.

- parse_dd_file
        Used when libmpopt_fuse is called with verify_dd=True.
"""

import os
import shutil
import subprocess
import tempfile
import time
import json
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

# == libmpopt ==
# Default binary name/solver. Can be overridden via env var or function arg.
DEFAULT_BIN    = os.environ.get("PPI_QAP_DD_FUSION_BIN",    "qap_dd_fusion")
DEFAULT_SOLVER = os.environ.get("PPI_QAP_DD_FUSION_SOLVER", "qpbo-i")


# ---------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------

# == libmpopt ==
def _resolve_binary(name: str) -> str:
    """Find `qap_dd_fusion` on PATH. Raise FileNotFoundError if missing."""
    if os.path.sep in name and os.path.isfile(name) and os.access(name, os.X_OK):
        return name
    located = shutil.which(name)
    if located is None:
        raise FileNotFoundError(
            f"qap_dd_fusion binary '{name}' not found on PATH. "
            f"Make sure the libmpopt conda env is active "
            f"(e.g. `conda activate fusion_moves`) or pass an absolute "
            f"path via qap_dd_fusion_bin=..."
        )
    return located


# ---------------------------------------------------------------
# .dd file writer  (Torresani / libmpopt iccv2021 format)
# ---------------------------------------------------------------
#   c <comment>
#   p <N0> <N1> <A> <E>      # left nodes, right nodes, #assignments, #edges
#   a <id> <i0> <i1> <cost>  # one assignment per line
#   e <a>  <b>  <cost>       # one edge per line (a, b are assignment ids)

# == libmpopt ==
def write_dd_file(path: str,
                  n_left: int, n_right: int,
                  assignments: List[Tuple[int, int, int, float]],
                  edges: List[Tuple[int, int, float]],
                  comment: Optional[str] = None) -> None:
    """Write a QAP model in libmpopt .dd format (Torresani et al.)."""
    with open(path, "w") as f:
        if comment:
            for line in str(comment).splitlines():
                f.write(f"c {line}\n")
        f.write(f"p {n_left} {n_right} {len(assignments)} {len(edges)}\n")
        for aid, left, right, cost in assignments:
            f.write(f"a {aid} {left} {right} {cost:.10g}\n")
        for a, b, cost in edges:
            f.write(f"e {a} {b} {cost:.10g}\n")


# == libmpopt ==
def write_solutions_file(path: str, solutions: List[np.ndarray]) -> None:
    """Write solutions in qap_dd_fusion's expected JSON-per-line format.

    parse_solutions() in dd_fusion.py calls json.loads() on each line and
    accesses obj['labeling'] and obj['energy'], so each line must be:
        {"labeling": [0, 1, 2, ...], "energy": <float>}

    energy=0.0 is a placeholder; qap_dd_fusion recomputes it internally.
    """
    with open(path, "w") as f:
        for sol in solutions:
            sol_list = np.asarray(sol, dtype=int).ravel().tolist()
            f.write(json.dumps({"labeling": sol_list, "energy": 0.0}) + "\n")


# ---------------------------------------------------------------
# .dd parser  (used only when verify_dd=True)
# ---------------------------------------------------------------

# == libmpopt ==
def parse_dd_file(path: str):
    """Parse a .dd file and return (no_left, no_right, assignments, edges).

    Used to self-check the file before invoking the binary (verify_dd=True).
    assignments: list of (id, left, right, cost)
    edges:       list of (a, b, cost)
    """
    no_left = no_right = no_assign = no_edges = None
    assignments, edges = [], []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("c") or ln.startswith("#"):
                continue
            toks = ln.split()
            tag = toks[0]
            if tag == "p":
                no_left, no_right = int(toks[1]), int(toks[2])
                no_assign, no_edges = int(toks[3]), int(toks[4])
            elif tag == "a":
                assignments.append(
                    (int(toks[1]), int(toks[2]), int(toks[3]), float(toks[4]))
                )
            elif tag == "e":
                edges.append((int(toks[1]), int(toks[2]), float(toks[3])))
    if no_left is None:
        raise ValueError(f"{path}: missing 'p' header")
    if no_assign != len(assignments):
        raise ValueError(
            f"{path}: header #assign={no_assign} but got {len(assignments)} 'a' lines"
        )
    if no_edges != len(edges):
        raise ValueError(
            f"{path}: header #edges={no_edges} but got {len(edges)} 'e' lines"
        )
    return no_left, no_right, assignments, edges


# ---------------------------------------------------------------
# Sparse .dd builder restricted to {p[i], q[i]} per left node
# ---------------------------------------------------------------

# == libmpopt ==
def build_qap_dd_for_fusion(
    A1, A2, K, lambda_param: float,
    current_sol: np.ndarray,
    candidate_sol: np.ndarray,
):
    """Construct a sparse QAP model for fusing two solutions.

    Energy convention
    -----------------
    libmpopt minimises; our PPI objective maximises:
        0.5 * sum(A1_sub * A2_sub) + lambda * sum_i K[i, sigma(i)]

    So all costs written to .dd are negated:
        unary     c(i, r)    = -lambda * K[i, r]
        pairwise  c(a, b)    = -A1[i,j] * A2[r_i, r_j]   for each A1 edge (i<j)
    """
    A1_np = np.asarray(A1)
    A2_np = np.asarray(A2)
    K_np  = np.asarray(K)
    n1, n2 = A1_np.shape[0], A2_np.shape[0]

    p = np.asarray(current_sol,   dtype=np.int64).ravel()
    q = np.asarray(candidate_sol, dtype=np.int64).ravel()
    if p.shape != (n1,) or q.shape != (n1,):
        raise ValueError(
            f"solution shape mismatch: expected ({n1},), got p={p.shape}, q={q.shape}"
        )

    # --- 1) Unary assignments: each left node gets labels {p[i], q[i]} ---
    assignments: List[Tuple[int, int, int, float]] = []
    assign_id: dict = {}   # (left, right) -> assignment id
    lam = float(lambda_param)
    for i in range(n1):
        for r in (int(p[i]), int(q[i])):
            if 0 <= r < n2 and (i, r) not in assign_id:
                aid = len(assignments)
                assignments.append((aid, i, r, -lam * float(K_np[i, r])))
                assign_id[(i, r)] = aid

    # --- 2) Pairwise edges: upper-triangle of A1 -------------------------
    edges: List[Tuple[int, int, float]] = []
    if sp.issparse(A1_np):
        coo = sp.triu(A1_np, k=1).tocoo()
        row, col, data = coo.row, coo.col, coo.data
    else:
        row, col = np.nonzero(np.triu(A1_np, k=1))
        data = A1_np[row, col]

    for idx in range(row.shape[0]):
        i, j = int(row[idx]), int(col[idx])
        a1_ij = float(data[idx])
        if a1_ij == 0.0:
            continue
        for ri in (int(p[i]), int(q[i])):
            if not (0 <= ri < n2):
                continue
            for rj in (int(p[j]), int(q[j])):
                if not (0 <= rj < n2) or ri == rj:
                    continue
                a2 = float(A2_np[ri, rj])
                if a2 == 0.0:
                    continue
                aid_a = assign_id[(i, ri)]
                aid_b = assign_id[(j, rj)]
                if aid_a > aid_b:
                    aid_a, aid_b = aid_b, aid_a
                edges.append((aid_a, aid_b, -a1_ij * a2))

    return n1, n2, assignments, edges


# ---------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------

# == libmpopt ==
def run_qap_dd_fusion(
    dd_path: str,
    solutions_path: str,
    output_path: str,
    qap_dd_fusion_bin: str = DEFAULT_BIN,
    solver: str = DEFAULT_SOLVER,
    timeout: float = 600.0,
    verbose: bool = True,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Invoke qap_dd_fusion and return (stdout, stderr).

    CLI: qap_dd_fusion [--solver <s>] INPUT.dd SOLUTIONS.txt --output OUT.txt
    """
    binp = _resolve_binary(qap_dd_fusion_bin)
    cmd  = [binp]
    if solver:
        cmd += ["--solver", str(solver)]
    cmd += [dd_path, solutions_path, "--output", output_path]
    if extra_args:
        cmd += list(extra_args)

    if verbose:
        print(f"  [libmpopt] CMD: {' '.join(cmd)}", flush=True)

    t0 = time.perf_counter()

    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    

    dt = time.perf_counter() - t0
    if verbose:
        print(f"  [libmpopt] elapsed={dt:.2f}s rc={proc.returncode}", flush=True)

    if proc.returncode != 0:
        try:
            hp = subprocess.run([binp, "--help"], capture_output=True,
                                text=True, timeout=5).stdout
        except Exception:
            hp = "<could not run --help>"
        raise RuntimeError(
            f"qap_dd_fusion failed (rc={proc.returncode}).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"------ stderr ------\n{proc.stderr[-2048:]}\n"
            f"------ stdout ------\n{proc.stdout[-2048:]}\n"
            f"------ --help  ------\n{hp[:2048]}"
        )
    return proc.stdout, proc.stderr


# ---------------------------------------------------------------
# High-level fuse function (drop-in replacement for asm_fusion_moves_qpbo)
# ---------------------------------------------------------------

# == libmpopt ==
def libmpopt_fuse(
    cfg,
    current_solution: np.ndarray,
    candidate_solution: np.ndarray,
    qap_dd_fusion_bin: str = DEFAULT_BIN,
    solver: str = DEFAULT_SOLVER,
    work_dir: Optional[str] = None,
    keep_files: bool = False,
    timeout: float = 600.0,
    verbose: bool = True,
    verify_dd: bool = False,
    extra_args: Optional[List[str]] = None,
) -> np.ndarray:
    """Fuse two PPI alignment solutions via libmpopt's qap_dd_fusion.

    Parameters
    ----------
    cfg : SimpleNamespace
        Must expose A1, A2, K, lambda_param (from asm_build_config).
    current_solution, candidate_solution : (n1,) np.ndarray of int
        Right-node index per left node; -1 for unassigned (dummy).
    qap_dd_fusion_bin : str
        Binary name or absolute path.
    solver : str
        --solver argument (default 'qpbo-i').
    work_dir : str or None
        Temp directory for .dd / solution files. Auto-cleaned unless
        keep_files=True.
    verify_dd : bool
        Re-parse the .dd file to catch format errors before subprocess call.
    extra_args : list[str] or None
        Extra flags forwarded verbatim to the subprocess.

    Returns
    -------
    np.ndarray of shape (n1,), dtype int64.
    """
    p  = np.asarray(current_solution,   dtype=np.int64).ravel()
    q  = np.asarray(candidate_solution, dtype=np.int64).ravel()
    n1 = cfg.A1.shape[0]
    n2 = cfg.A2.shape[0]

    # Fast path: identical solutions -> fusion is a no-op.
    if np.array_equal(p, q):
        if verbose:
            print("  [libmpopt] p == q -> skip fusion", flush=True)
        return p.copy()

    tmp_ctx = None
    if work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="libmpopt_fusion_")
        work_dir = tmp_ctx.name
    else:
        os.makedirs(work_dir, exist_ok=True)

    try:
        dd_path     = os.path.join(work_dir, "model.dd")
        sol_path    = os.path.join(work_dir, "primals.txt")
        output_path = os.path.join(work_dir, "fused.txt")

        # 1) Build sparse model restricted to labels {p[i], q[i]}.
        n1_chk, n2_chk, assignments, edges = build_qap_dd_for_fusion(
            cfg.A1, cfg.A2, cfg.K, cfg.lambda_param, p, q
        )
        assert n1_chk == n1 and n2_chk == n2
        write_dd_file(
            dd_path, n_left=n1, n_right=n2,
            assignments=assignments, edges=edges,
            comment="Auto-generated by libmpopt_fusion.py",
        )
        if verbose:
            print(f"  [libmpopt] .dd  n1={n1} n2={n2} "
                  f"#assignments={len(assignments)} #edges={len(edges)}", flush=True)

        # 2) Optional self-check.
        if verify_dd:
            nl, nr, asg, eg = parse_dd_file(dd_path)
            if (nl, nr, len(asg), len(eg)) != (n1, n2, len(assignments), len(edges)):
                raise RuntimeError(
                    f"parse_dd_file round-trip mismatch: "
                    f"wrote ({n1},{n2},{len(assignments)},{len(edges)}) "
                    f"but parsed ({nl},{nr},{len(asg)},{len(eg)})"
                )

        # 3) Write primals.
        write_solutions_file(sol_path, [p, q])

        # 4) Invoke binary.
        run_qap_dd_fusion(
            dd_path=dd_path,
            solutions_path=sol_path,
            output_path=output_path,
            qap_dd_fusion_bin=qap_dd_fusion_bin,
            solver=solver,
            timeout=timeout,
            verbose=verbose,
            extra_args=extra_args,
        )

        # 5) Read and validate fused solution.
        with open(output_path, "r") as fout:
            content = fout.read().strip()
        obj = json.loads(content)
        if isinstance(obj, dict):
            fused_list = obj.get("labeling") or obj.get("solution")
            if fused_list is None:
                raise ValueError(
                    f"qap_dd_fusion output JSON has no 'labeling'/'solution' key.\n"
                    f"Content: {content[:500]}"
                )
        elif isinstance(obj, list):
            fused_list = obj
        else:
            raise ValueError(
                f"Unsupported qap_dd_fusion output type: {type(obj).__name__}\n"
                f"Content: {content[:500]}"
            )

        fused = np.asarray(fused_list, dtype=np.int64).ravel()
        if fused.shape[0] != n1:
            raise ValueError(
                f"Output solution length {fused.shape[0]} != n1={n1}.\n"
                f"Content: {content[:500]}"
            )
        if np.any((fused < -1) | (fused >= n2)):
            raise ValueError(
                f"Output labels outside [-1, {n2-1}].\n"
                f"fused={fused}\nContent: {content[:500]}"
            )
        return fused

    finally:
        if tmp_ctx is not None and not keep_files:
            tmp_ctx.cleanup()
