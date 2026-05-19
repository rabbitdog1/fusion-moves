#!/usr/bin/env python3
"""
放置位置：<project_root>/run_batch_eval_species.py
运行命令：python3 run_batch_eval_species.py

项目结构：
  ./NETAL                        NETAL 可执行程序
  ./net_tab/*.tab                各物种 PPI 网络
  ./net_tab_seq/*.val            跨物种 biological similarity (仅作元信息记录)
  ./run_batch_eval_species.py    本脚本
  ./netal_runs/                  脚本运行时创建，存放 NETAL 的输入/输出

说明：
  本地 NETAL 要启用 biological similarity 需要同时提供
    net1-net1.val, net2-net2.val, net1-net2.val
  你只有跨物种 .val，缺自相似度 .val，所以本次 b=0（纯拓扑）。
  若以后补齐 self-similarity .val，把 USE_BIO_SIM=True, B_VALUE=0.5 即可。
"""

import os
import re
import stat
import shutil
import subprocess
import time
import glob
from openpyxl import Workbook


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NETAL_BIN = os.path.join(SCRIPT_DIR, "NETAL")
NET_TAB_DIR = os.path.join(SCRIPT_DIR, "net_tab")
NET_VAL_DIR = os.path.join(SCRIPT_DIR, "net_tab_seq")
RUN_DIR = os.path.join(SCRIPT_DIR, "netal_runs")

USE_BIO_SIM = False   # 缺 self-similarity .val，无法启用
B_VALUE = 0.0          # b=0 即 NETAL 默认

SPECIES_MAP = {
    "AT": "AThaliana",
    "CE": "CElegans",
    "DM": "DMelanogaster",
    "MM": "MMusculus",
    "RN": "RNorvegicus",
    "SP": "SPombe",
}

PAIRS = [
    ("AT", "DM"),
    ("CE", "AT"),
    ("CE", "DM"),
    ("CE", "MM"),
    ("MM", "AT"),
    ("MM", "DM"),
    ("RN", "AT"),
    ("RN", "CE"),
    ("RN", "DM"),
    ("RN", "MM"),
    ("RN", "SP"),
    ("SP", "AT"),
    ("SP", "CE"),
    ("SP", "DM"),
    ("SP", "MM"),
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
    return ICS, S3


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


def find_output_files(run_dir, tab1, tab2):
    prefix = f"({tab1}-{tab2})"
    alignment, evalfile = None, None
    for p in glob.glob(os.path.join(run_dir, prefix + "*")):
        if p.endswith(".alignment"):
            alignment = p
        elif p.endswith(".eval"):
            evalfile = p
    return alignment, evalfile


def ensure_executable(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"NETAL binary not found at: {path}")
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def stage_file(src, dst):
    """把源文件复制到目标路径（若已存在则跳过）。用 copy 而非 symlink 更稳。"""
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def main():
    print(f"Script dir : {SCRIPT_DIR}")
    print(f"NETAL bin  : {NETAL_BIN}")
    print(f"net_tab    : {NET_TAB_DIR}")
    print(f"net_val    : {NET_VAL_DIR}")
    print(f"Run dir    : {RUN_DIR}")
    print(f"Use bio sim: {USE_BIO_SIM}  (b={B_VALUE})")

    ensure_executable(NETAL_BIN)
    os.makedirs(RUN_DIR, exist_ok=True)

    rows = []

    for code1, code2 in PAIRS:
        sp1 = SPECIES_MAP[code1]
        sp2 = SPECIES_MAP[code2]
        tab1 = f"{sp1}.tab"
        tab2 = f"{sp2}.tab"

        src1 = os.path.join(NET_TAB_DIR, tab1)
        src2 = os.path.join(NET_TAB_DIR, tab2)
        val_name = f"{sp1}-{sp2}.val"
        val_src = os.path.join(NET_VAL_DIR, val_name)
        val_exists = os.path.exists(val_src)

        # Stage inputs
        stage_file(src1, os.path.join(RUN_DIR, tab1))
        stage_file(src2, os.path.join(RUN_DIR, tab2))
        if USE_BIO_SIM and val_exists:
            stage_file(val_src, os.path.join(RUN_DIR, val_name))

        # Build command
        cmd = [NETAL_BIN, tab1, tab2]
        if USE_BIO_SIM and val_exists:
            cmd += ["-b", str(B_VALUE)]

        print(f"\n=== {code1}-{code2}: {sp1} vs {sp2} ===")
        print(f"  cmd: {' '.join(cmd)}")
        t0 = time.time()
        subprocess.run(cmd, cwd=RUN_DIR, check=True)
        elapsed = time.time() - t0
        print(f"  Finished in {elapsed:.2f}s")

        alignment_path, eval_path = find_output_files(RUN_DIR, tab1, tab2)

        ec, nc = (None, None)
        if eval_path:
            ec, nc = parse_eval(eval_path)

        ICS, S3 = 0.0, 0.0
        if alignment_path:
            ICS, S3 = calc_ics_s3(src1, src2, alignment_path)

        print(f"  EC={ec}  NC={nc}  ICS={ICS:.6f}  S3={S3:.6f}")

        rows.append({
            "pair_code": f"{code1}-{code2}",
            "network1": tab1,
            "network2": tab2,
            "val_file": val_name if val_exists else "",
            "use_biological_similarity": USE_BIO_SIM and val_exists,
            "b_value": B_VALUE if (USE_BIO_SIM and val_exists) else 0.0,
            "runtime_seconds": round(elapsed, 3),
            "EC": ec,
            "NC": nc,
            "ICS": round(ICS, 6),
            "S3": round(S3, 6),
        })

    # Write Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "NETAL Species Results"
    headers = [
        "pair_code", "network1", "network2", "val_file",
        "use_biological_similarity", "b_value", "runtime_seconds",
        "EC", "NC", "ICS", "S3",
    ]
    ws.append(headers)
    for r in rows:
        ws.append([r[h] for h in headers])

    out_path = os.path.join(RUN_DIR, "netal_species_summary.xlsx")
    wb.save(out_path)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
