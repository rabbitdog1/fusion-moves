#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NETAL 候选解生成器:跑一次 NETAL,把 alignment 转成 n1 x n2 的 0/1 匹配矩阵。

公开 API:
    run_netal_matching_matrix(source_gw_path, target_gw_path, ...) -> np.ndarray
    parse_netal_alignment(alignment_path) -> Dict[str, str]
    alignment_to_matrix(mapping, source_nodes, target_nodes) -> np.ndarray
    find_netal_alignment_file(work_dir, source_tab_name, target_tab_name) -> str
"""
import glob
import os
import shutil
import stat
import subprocess
from typing import Dict, List

import numpy as np
import networkx


def parse_netal_alignment(alignment_path: str) -> Dict[str, str]:
    """
    解析 NETAL .alignment,兼容两种格式:
        nodeA -> nodeB
        nodeA<whitespace>nodeB
    """
    if not os.path.exists(alignment_path):
        raise FileNotFoundError(f"NETAL alignment file not found: {alignment_path}")

    mapping: Dict[str, str] = {}
    with open(alignment_path, 'r', encoding='utf-8') as f:
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


def alignment_to_matrix(mapping: Dict[str, str],
                        source_nodes: List,
                        target_nodes: List) -> np.ndarray:
    """
    严格按 source_nodes / target_nodes 的顺序构造 n1 x n2 矩阵。
    无法在节点表中找到的匹配对会被忽略并打印 matched 数量。
    """
    src2idx = {str(n): i for i, n in enumerate(source_nodes)}
    tgt2idx = {str(n): j for j, n in enumerate(target_nodes)}
    n1, n2 = len(source_nodes), len(target_nodes)

    M = np.zeros((n1, n2), dtype=np.float64)
    matched = 0
    for s, t in mapping.items():
        i = src2idx.get(str(s))
        j = tgt2idx.get(str(t))
        if i is None or j is None:
            continue
        M[i, j] = 1.0
        matched += 1
    print(f"[netal] alignment_to_matrix: n1={n1}, n2={n2}, "
          f"raw_pairs={len(mapping)}, matched={matched}")
    return M


def find_netal_alignment_file(work_dir: str,
                              source_tab_name: str,
                              target_tab_name: str) -> str:
    """
    NETAL 默认输出前缀: f"({source_tab_name}-{target_tab_name})"
    返回最新的 .alignment 文件路径。
    """
    prefix = f"({source_tab_name}-{target_tab_name})"
    pattern = os.path.join(work_dir, prefix + "*.alignment")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No NETAL alignment file matching pattern: {pattern}"
        )
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _ensure_executable(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NETAL binary not found at: {path}")
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _stage_file(src: str, dst: str) -> None:
    if not os.path.exists(src):
        raise FileNotFoundError(f"Required input file not found: {src}")
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def _gw_to_tab_path(gw_path: str) -> str:
    """约定: .gw 与 .tab 同名同目录,扩展名不同。"""
    base, _ = os.path.splitext(gw_path)
    return base + ".tab"


def run_netal_matching_matrix(source_gw_path: str,
                              target_gw_path: str,
                              netal_bin: str = "./Algorithms/Netal/NETAL",
                              work_dir: str = "./netal_runs",
                              force_rerun: bool = False) -> np.ndarray:
    """
    在 (source, target) 上跑一次 NETAL,返回 n1 x n2 的 0/1 匹配矩阵 M。
    要求每个 .gw 在同目录下已存在同名 .tab(由 gw_to_tab.py 生成)。
    """
    # 1. 读图,固定节点顺序(与 ppikktetm.py 中 networkx.to_numpy_array 顺序一致)
    source_graph = networkx.read_leda(source_gw_path)
    target_graph = networkx.read_leda(target_gw_path)
    source_nodes = list(source_graph.nodes())
    target_nodes = list(target_graph.nodes())

    # 2. 同名 .tab
    source_tab_path = _gw_to_tab_path(source_gw_path)
    target_tab_path = _gw_to_tab_path(target_gw_path)
    if not os.path.exists(source_tab_path):
        raise FileNotFoundError(
            f".tab not found for source: {source_tab_path}. "
            f"请先运行 gw_to_tab.py 生成 .tab"
        )
    if not os.path.exists(target_tab_path):
        raise FileNotFoundError(
            f".tab not found for target: {target_tab_path}. "
            f"请先运行 gw_to_tab.py 生成 .tab"
        )

    source_tab_name = os.path.basename(source_tab_path)
    target_tab_name = os.path.basename(target_tab_path)

    # 3. 准备 work_dir
    os.makedirs(work_dir, exist_ok=True)
    _stage_file(source_tab_path, os.path.join(work_dir, source_tab_name))
    _stage_file(target_tab_path, os.path.join(work_dir, target_tab_name))

    # 4. 复用已有 alignment(若存在且未要求重跑)
    prefix = f"({source_tab_name}-{target_tab_name})"
    cached = glob.glob(os.path.join(work_dir, prefix + "*.alignment"))
    if cached and not force_rerun:
        cached.sort(key=os.path.getmtime, reverse=True)
        alignment_path = cached[0]
        print(f"[netal] reuse cached alignment: {alignment_path}")
    else:
        netal_bin_abs = os.path.abspath(netal_bin)
        _ensure_executable(netal_bin_abs)
        cmd = [netal_bin_abs, source_tab_name, target_tab_name]
        print(f"[netal] run: {' '.join(cmd)} (cwd={work_dir})")
        subprocess.run(cmd, cwd=work_dir, check=True)
        alignment_path = find_netal_alignment_file(
            work_dir, source_tab_name, target_tab_name
        )
        print(f"[netal] alignment file: {alignment_path}")

    # 5. 解析 alignment 转为匹配矩阵
    mapping = parse_netal_alignment(alignment_path)
    M = alignment_to_matrix(mapping, source_nodes, target_nodes)
    return M
