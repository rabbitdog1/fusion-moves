#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_wave_noise_batch.py

批量运行 WAVE，对 syeast0 与 syeast{05,10,15,20,25} 做网络对齐，
统计运行时间与 NC / EC / ICS / S3 指标，结果写入 xlsx。

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

from openpyxl import Workbook
# 可以单独生成一个 uniform.sim
# n = 1004
# with open("noise_lst/syeast_uniform.sim", "w") as f:
#     for i in range(n):
#         for j in range(n):
#             f.write(f"{i} {j} 1.0\n")



# ============ 配置区 ============
PROJECT_ROOT = Path(__file__).resolve().parent
WAVE_BIN = PROJECT_ROOT / "WAVE"
NOISE_DIR = PROJECT_ROOT / "noise_lst"
OUTPUT_DIR = PROJECT_ROOT / "output"

BASE_NET = "syeast0.lst"
NOISE_NETS = [
    "syeast05.lst",
    "syeast10.lst",
    "syeast15.lst",
    "syeast20.lst",
    "syeast25.lst",
]
SIM_FILE = NOISE_DIR / "syeast_uniform.sim"

OUTPUT_XLSX = OUTPUT_DIR / "wave_noise_results.xlsx"
# ================================


def read_lst(path: Path):
    """读取 lst 格式网络，返回 (n, edges_set)，edges_set 中的边以 (min,max) 元组存储。"""
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
    """读取 alignment 文件，返回 dict: source_node -> target_node"""
    f_map = {}
    with path.open("r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            s, t = int(parts[0]), int(parts[1])
            f_map[s] = t
    return f_map


def compute_metrics(g1_edges, g2_edges, f_map, n1):
    """计算 NC / EC / ICS / S3。"""
    # NC：identity mapping 为真实映射
    correct = sum(1 for s, t in f_map.items() if s == t)
    nc = correct / n1 if n1 > 0 else 0.0

    # conserved edges
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

    # induced target edges：G2 中两端点都被映射到的边数
    mapped_targets = set(f_map.values())
    induced = 0
    for (a, b) in g2_edges:
        if a in mapped_targets and b in mapped_targets:
            induced += 1

    e1 = len(g1_edges)
    ec = conserved / e1 if e1 > 0 else 0.0
    ics = conserved / induced if induced > 0 else 0.0
    denom_s3 = e1 + induced - conserved
    s3 = conserved / denom_s3 if denom_s3 > 0 else 0.0

    return nc, ec, ics, s3


def run_wave(net1: Path, net2: Path, sim: Path, out_aln: Path):
    """调用 WAVE，返回 wall-clock 耗时(秒)。失败抛出异常。"""
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
    # 基础检查
    if not WAVE_BIN.exists():
        raise FileNotFoundError(f"未找到 WAVE 可执行文件: {WAVE_BIN}")
    if not os.access(str(WAVE_BIN), os.X_OK):
        raise PermissionError(f"WAVE 没有可执行权限: {WAVE_BIN} (请先 chmod +x)")
    if not SIM_FILE.exists():
        raise FileNotFoundError(f"未找到 similarity 文件: {SIM_FILE}")
    base_path = NOISE_DIR / BASE_NET
    if not base_path.exists():
        raise FileNotFoundError(f"未找到基础网络文件: {base_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 预读 G1
    n1, g1_edges = read_lst(base_path)

    results = []

    for noise_net in NOISE_NETS:
        net2_path = NOISE_DIR / noise_net
        if not net2_path.exists():
            raise FileNotFoundError(f"未找到目标网络文件: {net2_path}")

        pair_tag = f"{base_path.stem}_vs_{net2_path.stem}"
        out_aln = OUTPUT_DIR / f"{pair_tag}.aln"

        try:
            elapsed = run_wave(base_path, net2_path, SIM_FILE, out_aln)
        except Exception as e:
            raise RuntimeError(f"[{pair_tag}] WAVE 运行失败: {e}") from e

        try:
            n2, g2_edges = read_lst(net2_path)
            f_map = read_alignment(out_aln)
            nc, ec, ics, s3 = compute_metrics(g1_edges, g2_edges, f_map, n1)
        except Exception as e:
            raise RuntimeError(f"[{pair_tag}] 指标计算失败: {e}") from e

        results.append({
            "pair": pair_tag,
            "time_seconds": elapsed,
            "NC": nc,
            "EC": ec,
            "ICS": ics,
            "S3": s3,
        })
        print(f"[OK] {pair_tag}  time={elapsed:.3f}s  "
              f"NC={nc:.4f}  EC={ec:.4f}  ICS={ics:.4f}  S3={s3:.4f}")

    # 写 xlsx
    wb = Workbook()
    ws = wb.active
    ws.title = "wave_results"
    headers = ["pair", "time_seconds", "NC", "EC", "ICS", "S3"]
    ws.append(headers)
    for r in results:
        ws.append([r[h] for h in headers])
    wb.save(str(OUTPUT_XLSX))
    print(f"\n结果已写入: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
