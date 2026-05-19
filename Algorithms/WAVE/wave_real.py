#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_wave_cross_species_batch.py

批量运行 WAVE，对真实跨物种 PPI 网络进行对齐，统计运行时间与 EC / ICS / S3
指标（NC 输出为 "NA"，因无 ground-truth ortholog mapping），结果写入 xlsx。

关于 WAVE 的命令行参数：
  根据项目根目录下 readme.txt 的说明以及 WAVE 可执行文件打印的 usage：
      Usage: ./WAVE [input-network1] [input-network2] [input-similarity] [output-alignment]
  当前版本未发现其他可设置参数，因此程序仅按 4 个位置参数调用 WAVE。
"""

import os
import sys
import time
import subprocess
from pathlib import Path

try:
    from openpyxl import Workbook
except ImportError:
    sys.stderr.write(
        "错误：缺少依赖 openpyxl。请先安装：\n"
        "    pip install openpyxl\n"
    )
    sys.exit(1)


# ============ 配置区 ============
PROJECT_ROOT = Path(__file__).resolve().parent
WAVE_BIN = PROJECT_ROOT / "WAVE"
NET_DIR = PROJECT_ROOT / "net_lst"
SIM_DIR = PROJECT_ROOT / "net_lst_seq"
OUTPUT_DIR = PROJECT_ROOT / "output"

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

OUTPUT_XLSX = OUTPUT_DIR / "wave_cross_species_results.xlsx"
# ================================


def read_lst(path: Path):
    """读取 lst 格式网络，返回 (n, edges_set)，边以 (min, max) 元组存储。"""
    with path.open("r") as f:
        first = f.readline().split()
        n, m = int(first[0]), int(first[1])
        edges = set()
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    return n, edges


def read_alignment(path: Path):
    """读取 alignment 文件，返回 dict: source_node -> target_node。"""
    f_map = {}
    with path.open("r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            s, t = int(parts[0]), int(parts[1])
            f_map[s] = t
    return f_map


def compute_metrics(g1_edges, g2_edges, f_map):
    """计算 EC / ICS / S3 与辅助计数。NC 在此不计算（跨物种数据无 identity truth）。"""
    conserved = 0
    for (u, v) in g1_edges:
        if u not in f_map or v not in f_map:
            continue
        fu, fv = f_map[u], f_map[v]
        if fu == fv:
            continue
        a, b = (fu, fv) if fu < fv else (fv, fu)
        if (a, b) in g2_edges:
            conserved += 1

    mapped_targets = set(f_map.values())
    induced = sum(
        1 for (a, b) in g2_edges if a in mapped_targets and b in mapped_targets
    )

    e1 = len(g1_edges)
    ec = conserved / e1 if e1 > 0 else 0.0
    ics = conserved / induced if induced > 0 else 0.0
    denom_s3 = e1 + induced - conserved
    s3 = conserved / denom_s3 if denom_s3 > 0 else 0.0

    return ec, ics, s3, conserved, induced


def run_wave(net1: Path, net2: Path, sim: Path, out_aln: Path):
    """调用 WAVE，返回 wall-clock 耗时(秒)。失败抛异常。"""
    cmd = [str(WAVE_BIN), str(net1), str(net2), str(sim), str(out_aln)]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"WAVE 运行失败 (returncode={proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {proc.stdout.decode(errors='replace')}\n"
            f"stderr: {proc.stderr.decode(errors='replace')}"
        )
    if not out_aln.exists():
        raise RuntimeError(f"WAVE 运行完毕但未生成 alignment 文件: {out_aln}")
    return elapsed


def main():
    if not WAVE_BIN.exists():
        raise FileNotFoundError(f"未找到 WAVE 可执行文件: {WAVE_BIN}")
    if not os.access(str(WAVE_BIN), os.X_OK):
        raise PermissionError(f"WAVE 没有可执行权限: {WAVE_BIN} (请先 chmod +x)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for code1, code2 in PAIRS:
        if code1 not in SPECIES_MAP or code2 not in SPECIES_MAP:
            raise KeyError(f"未知物种缩写: ({code1}, {code2})")
        sp1 = SPECIES_MAP[code1]
        sp2 = SPECIES_MAP[code2]
        pair_code = f"{code1}-{code2}"
        pair_tag = f"{sp1}_vs_{sp2}"

        net1_path = NET_DIR / f"{sp1}.lst"
        net2_path = NET_DIR / f"{sp2}.lst"
        sim_path = SIM_DIR / f"{sp1}-{sp2}.sim"
        out_aln = OUTPUT_DIR / f"{pair_tag}.aln"

        if not net1_path.exists():
            raise FileNotFoundError(f"[{pair_tag}] 未找到源网络文件: {net1_path}")
        if not net2_path.exists():
            raise FileNotFoundError(f"[{pair_tag}] 未找到目标网络文件: {net2_path}")
        if not sim_path.exists():
            raise FileNotFoundError(f"[{pair_tag}] 未找到 similarity 文件: {sim_path}")

        try:
            elapsed = run_wave(net1_path, net2_path, sim_path, out_aln)
        except Exception as e:
            raise RuntimeError(f"[{pair_tag}] WAVE 运行失败: {e}") from e

        try:
            n1, g1_edges = read_lst(net1_path)
            n2, g2_edges = read_lst(net2_path)
            f_map = read_alignment(out_aln)
            ec, ics, s3, conserved, induced = compute_metrics(g1_edges, g2_edges, f_map)
        except Exception as e:
            raise RuntimeError(f"[{pair_tag}] 指标计算失败: {e}") from e

        results.append({
            "pair_code": pair_code,
            "species1": sp1,
            "species2": sp2,
            "source_network": net1_path.name,
            "target_network": net2_path.name,
            "similarity_file": sim_path.name,
            "alignment_file": out_aln.name,
            "time_seconds": elapsed,
            "NC": "NA",
            "EC": ec,
            "ICS": ics,
            "S3": s3,
            "source_nodes": n1,
            "target_nodes": n2,
            "source_edges": len(g1_edges),
            "target_edges": len(g2_edges),
            "conserved_edges": conserved,
            "induced_target_edges": induced,
        })
        print(f"[OK] {pair_tag}  time={elapsed:.6f}s  "
              f"EC={ec:.4f}  ICS={ics:.4f}  S3={s3:.4f}  NC=NA")

    wb = Workbook()
    ws = wb.active
    ws.title = "wave_cross_species"
    headers = [
        "pair_code",
        "species1",
        "species2",
        "source_network",
        "target_network",
        "similarity_file",
        "alignment_file",
        "time_seconds",
        "NC",
        "EC",
        "ICS",
        "S3",
        "source_nodes",
        "target_nodes",
        "source_edges",
        "target_edges",
        "conserved_edges",
        "induced_target_edges",
    ]
    ws.append(headers)
    for r in results:
        ws.append([r[h] for h in headers])
    wb.save(str(OUTPUT_XLSX))
    print(f"\n结果已写入: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
