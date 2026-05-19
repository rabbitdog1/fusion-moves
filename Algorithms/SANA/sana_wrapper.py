#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SANA wrapper: 在一对 .gw 网络上运行 SANA，返回 n1 x n2 的匹配矩阵
以及对应的 candidate_solution（行=source 节点顺序，列=target 节点顺序）。
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx


def parse_sana_alignment(alignment_path: str) -> Dict[str, str]:
    """
    解析 SANA 输出的 .align 文件。
    每行: '<source_node>\t<target_node>'（或空格分隔）。
    返回 dict: source 节点名 -> target 节点名。
    """
    mapping: Dict[str, str] = {}
    with open(alignment_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = parts[1]
    return mapping


def alignment_to_matrix(mapping: Dict[str, str],
                        source_nodes: List[str],
                        target_nodes: List[str]) -> np.ndarray:
    """
    从 name->name 映射构建 n1 x n2 的 0/1 匹配矩阵。
    无法识别的节点名直接忽略，重复占用的 source/target 也忽略。
    """
    n1, n2 = len(source_nodes), len(target_nodes)
    src2idx = {str(n): i for i, n in enumerate(source_nodes)}
    tgt2idx = {str(n): j for j, n in enumerate(target_nodes)}

    M = np.zeros((n1, n2), dtype=np.float64)
    used_tgt = set()
    matched = 0
    for s, t in mapping.items():
        i = src2idx.get(str(s))
        j = tgt2idx.get(str(t))
        if i is None or j is None:
            continue
        if M[i].sum() > 0:        # 该 source 已分配
            continue
        if j in used_tgt:         # 该 target 已被占用
            continue
        M[i, j] = 1.0
        used_tgt.add(j)
        matched += 1

    print(f"[SANA] matched {matched}/{n1} source nodes "
          f"(target candidates {n2}); raw mapping size = {len(mapping)}")
    return M


def run_sana_matching_matrix(
    source_gw_path: str,
    target_gw_path: str,
    sana_bin: str = "./Algorithms/SANA/sana_main.py",
    work_dir: str = "./sana_runs",
    runtime_minutes: float = 10.0,
    objective_flag: str = "-ec",
    objective_weight: str = "1",
    extra_args: List[str] = None,
    force_rerun: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 source_gw_path -> target_gw_path 上运行 SANA，返回:
        M:                  shape (n1, n2) 的 0/1 匹配矩阵
        candidate_solution: shape (n1,)，solution[i] = j 或 -1（未匹配）
    """
    os.makedirs(work_dir, exist_ok=True)

    src_stem = Path(source_gw_path).stem
    tgt_stem = Path(target_gw_path).stem

    #new====================
    # 将 SANA 关键参数写入输出文件名，避免不同实验复用同一个 .align
    safe_objective_flag = objective_flag.replace("-", "")
    safe_objective_weight = str(objective_weight).replace(".", "p")
    safe_runtime = str(runtime_minutes).replace(".", "p")

    out_prefix = os.path.join(
        work_dir,
        f"{src_stem}__{tgt_stem}__{safe_objective_flag}{safe_objective_weight}__t{safe_runtime}"
    )
    #new====================
    # out_prefix = os.path.join(work_dir, f"{src_stem}__{tgt_stem}")
    align_path = out_prefix + ".align"

    # 读取节点顺序，必须与 SANA 看到的一致
    source_graph = nx.read_leda(source_gw_path)
    target_graph = nx.read_leda(target_gw_path)
    source_nodes = list(source_graph.nodes())
    target_nodes = list(target_graph.nodes())

    need_run = force_rerun or (not os.path.exists(align_path))
    if need_run:
        cmd = [
            sana_bin,
            "-fg1", source_gw_path,
            "-fg2", target_gw_path,
            objective_flag, objective_weight,
            "-t", str(runtime_minutes),
            "-tolerance", "0",
            "-o", out_prefix,
        ]
        if extra_args:
            cmd.extend(extra_args)

        print(f"[SANA] running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"无法启动 SANA: '{sana_bin}' ({e})"
            )

        if proc.returncode != 0:
            print(proc.stdout)
            raise RuntimeError(
                f"SANA 退出码非零: {proc.returncode}. "
                f"请检查 sana_bin 路径与命令行参数。"
            )

    if not os.path.exists(align_path):
        raise FileNotFoundError(
            f"未找到 SANA 输出的 alignment 文件: {align_path}\n"
            f"请确认 sana_bin 是否能产生 '<out_prefix>.align'。"
        )

    mapping = parse_sana_alignment(align_path)
    M = alignment_to_matrix(mapping, source_nodes, target_nodes)

    n1 = M.shape[0]
    candidate_solution = np.full(n1, -1, dtype=int)
    rows, cols = np.where(M == 1)
    candidate_solution[rows] = cols

    return M, candidate_solution
