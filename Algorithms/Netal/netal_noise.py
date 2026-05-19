#!/usr/bin/env python3
"""
放置位置：~/network_alignment/Netal/run_batch_eval.py
运行命令：cd ~/network_alignment/Netal && python3 run_batch_eval.py
"""

import os
import re
import subprocess
import time
import glob
from openpyxl import Workbook


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
NETAL_BIN = os.path.join(SCRIPT_DIR, "NETAL")

PAIRS = [
    ("syeast0.tab", "syeast05.tab"),
    ("syeast0.tab", "syeast10.tab"),
    ("syeast0.tab", "syeast15.tab"),
    ("syeast0.tab", "syeast20.tab"),
    ("syeast0.tab", "syeast25.tab"),
]


def load_edges(filepath):
    edges = set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u == v:
                continue
            edges.add((min(u, v), max(u, v)))
    return edges


def load_alignment(filepath):
    mapping = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "->":
                mapping[parts[0]] = parts[2]
            elif len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def calc_ics_s3(g1_path, g2_path, alignment_path):
    E1 = load_edges(g1_path)
    E2 = load_edges(g2_path)
    f = load_alignment(alignment_path)

    C = 0
    for u, v in E1:
        if u in f and v in f:
            e = (min(f[u], f[v]), max(f[u], f[v]))
            if e in E2:
                C += 1

    mapped_nodes = set(f.values())
    E2_prime = sum(1 for u, v in E2 if u in mapped_nodes and v in mapped_nodes)

    ICS = C / E2_prime if E2_prime else 0.0
    denom = len(E1) + E2_prime - C
    S3 = C / denom if denom else 0.0
    return C, len(E1), E2_prime, ICS, S3


def parse_eval(eval_path):
    ec, nc = None, None
    with open(eval_path) as f:
        for line in f:
            m = re.search(r"Edge Correctness\s*:\s*([\d.]+)", line)
            if m:
                ec = float(m.group(1))
            m = re.search(r"Node Correctness\s*:\s*([\d.]+)", line)
            if m:
                nc = float(m.group(1))
    return ec, nc


def find_output_files(g1, g2):
    """Find the .alignment and .eval files NETAL generated for this pair."""
    prefix = f"({g1}-{g2})"
    alignment = None
    evalfile = None
    for p in glob.glob(os.path.join(RESULTS_DIR, prefix + "*")):
        if p.endswith(".alignment"):
            alignment = p
        elif p.endswith(".eval"):
            evalfile = p
    return alignment, evalfile


def main():
    rows = []

    for g1_name, g2_name in PAIRS:
        g1_path = os.path.join(RESULTS_DIR, g1_name)
        g2_path = os.path.join(RESULTS_DIR, g2_name)

        print(f"=== Running: {g1_name} vs {g2_name} ===")
        t0 = time.time()
        subprocess.run(
            [NETAL_BIN, g1_name, g2_name],
            cwd=RESULTS_DIR,
            check=True,
        )
        elapsed = time.time() - t0
        print(f"  Finished in {elapsed:.2f}s")

        alignment_path, eval_path = find_output_files(g1_name, g2_name)

        ec, nc = (None, None)
        if eval_path:
            ec, nc = parse_eval(eval_path)

        ICS, S3 = 0.0, 0.0
        if alignment_path:
            _, _, _, ICS, S3 = calc_ics_s3(g1_path, g2_path, alignment_path)

        print(f"  EC={ec}  NC={nc}  ICS={ICS:.6f}  S3={S3:.6f}")
        rows.append({
            "network1": g1_name,
            "network2": g2_name,
            "runtime_seconds": round(elapsed, 3),
            "EC": ec,
            "NC": nc,
            "ICS": round(ICS, 6),
            "S3": round(S3, 6),
        })

    # Write Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "NETAL Results"
    headers = ["network1", "network2", "runtime_seconds", "EC", "NC", "ICS", "S3"]
    ws.append(headers)
    for r in rows:
        ws.append([r[h] for h in headers])

    out_path = os.path.join(RESULTS_DIR, "netal_summary.xlsx")
    wb.save(out_path)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
