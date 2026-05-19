#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPI Network Alignment 批量评测脚本 (ETM+ASM 版本)
基于 f2.ipynb 整合为单个 .py 文件，支持 15 个物种配对 × 3 个 a 值的批量运行。
用法: python ppi_batch_etm.py
"""

import gc
import os
import time
import math
import types
from typing import Tuple, List, Dict

import networkx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
import thinqpbo as tq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from line_profiler import LineProfiler
import ot
import pygmtools as pygm
import pandas as pd
#change=============
from Algorithms.Netal.netal import run_netal_matching_matrix
from Algorithms.SANA.sana_wrapper import run_sana_matching_matrix

#change=============


device = torch.device("cuda")

###############################################################################
# 基础工具函数
###############################################################################

def compute_alpha(M, delta_f, D, adjacency1, adjacency2, K=None, lamb=1, device='cuda'):
    """GPU 版本的 compute_alpha"""
    def tr_dot_torch(a, b):
        return torch.sum(a * b.T)

    tr_matrix1 = tr_dot_torch(M.T, delta_f)
    tr_matrix2 = tr_dot_torch(delta_f, D.T)
    tr_matrix3 = tr_matrix2
    tr_matrix4 = tr_dot_torch(D.T * adjacency1, D * adjacency2)

    if K is None or K.max() == 0:
        tr_matrix5 = 0
        tr_matrix6 = 0
    else:
        tr_matrix5 = tr_dot_torch(D.T, K)
        tr_matrix6 = tr_dot_torch(M.T, K)

    tr_matrix1m2m3 = tr_matrix1 - tr_matrix2 - tr_matrix3
    alpha_a = tr_matrix1m2m3 + tr_matrix4
    alpha_b = -tr_matrix1m2m3 - tr_matrix1 + lamb * tr_matrix5 - lamb * tr_matrix6
    alpha_op = -alpha_b / (2 * alpha_a)
    return alpha_op.item()

def tr_dot(a, b):
    trace = np.sum(np.multiply(a, b.T))
    return trace

def hugarian_gpu(matrix, device='cuda'):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32, device=device)
    else:
        matrix = matrix.to(device=device)
    matrix_batch = matrix.unsqueeze(0)
    P = pygm.hungarian(matrix_batch, backend='pytorch')
    return P.squeeze(0)

def hugarian(matrix):
    n, m = matrix.shape
    P = np.mat(np.zeros((n, m)))
    row_ind, col_ind = linear_sum_assignment(-matrix)
    P[row_ind, col_ind] = 1
    return P

###############################################################################
# ETM
###############################################################################

def uot_helper(benefit_mat, tau_a, tau_b, eps=0.01, epoch_iter=500,tol = 0.1, device='cuda', dtype=torch.float32):
    if isinstance(benefit_mat, torch.Tensor):
        benefit_mat = benefit_mat.to(dtype=dtype, device=device)
    else:
        benefit_mat = torch.tensor(np.asarray(benefit_mat), dtype=dtype, device=device)

    n, m = benefit_mat.shape

    # ---- 收益 → 归一化代价 [0, 1] ----
    D_max = benefit_mat.max()
    D_min = benefit_mat.min()
    cost_range = D_max - D_min
    if cost_range < 1e-12:
        # 收益矩阵几乎恒定 → 无法区分好坏匹配 → 返回零 slack
        zeros_n = torch.zeros(n, dtype=dtype, device=device)
        zeros_m = torch.zeros(m, dtype=dtype, device=device)
        zeros_nm = torch.zeros(n, m, dtype=dtype, device=device)
        return zeros_n, zeros_m, 0.0, zeros_nm

    cost_mat = (D_max - benefit_mat) / cost_range  # 归一化到 [0, 1]

    # === 【修复 A】矩形匹配的 marginal 质量守恒 ===
    # 注入式匹配：总质量必须相等，否则 Sinkhorn/UOT 不收敛到合法 plan
    if n <= m:
        src_weight = torch.ones(n, dtype=dtype, device=device)
        trg_weight = torch.ones(m, dtype=dtype, device=device) * (float(n) / float(m))
    else:
        src_weight = torch.ones(n, dtype=dtype, device=device) * (float(m) / float(n))
        trg_weight = torch.ones(m, dtype=dtype, device=device)

    # ---- 初始化对偶变量 ----
    zeta = torch.tensor(0.0, dtype=dtype, device=device)
    u = torch.ones(n, dtype=dtype, device=device) / n
    v = torch.ones(m, dtype=dtype, device=device) / m

    # ---- ETM 迭代 (与原始 uot_helper 结构一致) ----
    for e in range(epoch_iter):
        prev_u = u
        prev_v = v

        # ---- 【修复】用 log-sum-exp 替代直接 exp，防止溢出 ----
        # 原式: den = sum_i exp((u[i] - cost_mat[i,j]) / eps)
        exponent_den = (u.reshape(-1, 1) - cost_mat) / eps          # (n, m)
        max_den = exponent_den.max(dim=0).values                     # (m,)
        den = torch.exp(max_den) * torch.sum(
            torch.exp(exponent_den - max_den.unsqueeze(0)), dim=0)   # (m,)
        # den = torch.sum(torch.exp((u.reshape(-1, 1) - cost_mat) / eps), axis=0) # u.unsqueeze(1) 
        den = torch.clamp(den, min=1e-30) # 防止除零

        # term = torch.exp(-cost_mat / eps) / den
        # 用 log 域计算避免上溢
        log_term = -cost_mat / eps - torch.log(den).unsqueeze(0)     # (n, m)
        log_term = torch.clamp(log_term, min=-80.0, max=80.0)
        term = torch.exp(log_term)

        new_trg_weight = trg_weight * torch.exp(torch.clamp(-v / tau_b, -80, 80))

        sum_term = torch.sum(term * new_trg_weight.reshape(1, -1), axis=1)
        sum_term = torch.clamp(sum_term, min=1e-30)

        u = (tau_a * eps / (tau_a + eps)) * (
            torch.log(src_weight * torch.exp(torch.clamp(-zeta / tau_a, min=-80.0, max=80.0)))
            - torch.log(sum_term)
        )

        exponent_v = (u.unsqueeze(1) - cost_mat) / eps               # (n, m)
        max_v = exponent_v.max(dim=0).values                          # (m,)
        log_sum_v = max_v + torch.log(torch.clamp(
            torch.sum(torch.exp(exponent_v - max_v.unsqueeze(0)), dim=0),
            min=1e-30))
        v = -eps * log_sum_v


        if torch.linalg.norm(prev_u - u) <= tol and torch.linalg.norm(prev_v - v) <= tol:
            print('Using Epoch to converge:', e)
            break
        
    # ---- 更新 ζ (质量平衡) ----
    exp_u = torch.exp(torch.clamp(-u / tau_a, min=-80.0, max=80.0))
    exp_v = torch.exp(torch.clamp(-v / tau_b, min=-80.0, max=80.0))
    src_weight_sum = torch.sum(src_weight * exp_u)
    trg_weight_sum = torch.sum(trg_weight * exp_v)
    kappa = (tau_a * tau_b) / (tau_a + tau_b)
    ratio = src_weight_sum / torch.clamp(trg_weight_sum, min=1e-30)
    # zeta = kappa * torch.log(src_weight_sum / trg_weight_sum)
    zeta = kappa * torch.log(torch.clamp(ratio, min=1e-30))

    # ---- Slack 变量 (在归一化代价空间中) ----
    s_cost = torch.clamp( cost_mat - u.reshape(-1, 1) - v.reshape(1, -1), min=0.0 )  # 负值不惩罚

    s_benefit = s_cost * cost_range

    return u, v, zeta, s_benefit


###############################################################################
# ETM+ASM GPU
###############################################################################

profiler = LineProfiler()

@profiler
def adaptive_softassign_torch(matrix, gamma_ini=1, eps=0.04, device='cuda', dtype=torch.float32, use_amp=False,
                              row_target=None, col_target=None):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=dtype, device=device)
    else:
        matrix = matrix.to(dtype=dtype, device=device)

    matrix = matrix / matrix.max()
    n, m = matrix.shape
    print(f'dim={n},{m}')

    if row_target is None:
        row_target = torch.ones(n, dtype=dtype, device=device)
    if col_target is None:
        col_target = torch.ones(m, dtype=dtype, device=device)

    P = torch.ones((n, m), dtype=dtype, device=device) / n
    # beta = math.log(n) * gamma_ini
    beta = torch.log(torch.tensor(float(n), dtype=dtype, device=device)) * gamma_ini
    K = torch.exp(matrix * beta)

    v = torch.ones(m, dtype=dtype, device=device)
    u = torch.ones(n, dtype=dtype, device=device)

    diff = 10.0

    while diff > eps:
        Q = K * P
        iter_sk = 1
        res_diff = 1.0

        while res_diff > 0.2:
            # u_new = row_target / torch.matmul(Q, v)
            # v_new = col_target / torch.matmul(Q.T, u_new)
            u_new = row_target / torch.clamp(torch.matmul(Q, v), min=1e-30)
            v_new = col_target / torch.clamp(torch.matmul(Q.T, u_new), min=1e-30)

            iter_sk += 1

            if iter_sk % 3 == 1 and iter_sk > 3:
                norm_u = u_new / u_new.max()
                norm_v = v_new / v_new.max()
                norm_u_prev = u / u.max()
                norm_v_prev = v / v.max()
                res_diff = torch.norm(norm_u - norm_u_prev, p=1) + torch.norm(norm_v - norm_v_prev, p=1)

            u = u_new
            v = v_new

        P1 = torch.outer(u, v) * Q
        diff = torch.max(torch.sum(torch.abs(P - P1), dim=0))
        P = P1
        gamma_ini += 1

    return P, gamma_ini


def graphmatch_ASM_torch(
    adjacency1, adjacency2, K=0, tol=0.1, alpha=1, lamb=1, gamma0=1, adaptive_alpha=0, niter_max=30,
    X_init_normalized=None,
    tau=0.1, eps_uot=0.01, lambda_mrot=11, uot_iters=10,
    device='cuda', precision='float32', method='asm'
):
    precision = precision.lower()
    use_amp = False

    if precision == 'float64':
        dtype = torch.float64
    elif precision == 'float16':
        dtype = torch.float16
    elif precision == 'tf32':
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif precision == 'amp':
        use_amp = True
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype = torch.float32

    device = torch.device(device)
    starttime = time.perf_counter()

    def to_tensor(x):
        return torch.tensor(x, dtype=dtype, device=device) if not isinstance(x, torch.Tensor) else x.to(dtype=dtype, device=device)

    adjacency1 = to_tensor(adjacency1)
    adjacency2 = to_tensor(adjacency2)
    K = to_tensor(K)

    n, _ = adjacency1.shape
    m, _ = adjacency2.shape

    if X_init_normalized is not None:
        N = to_tensor(X_init_normalized)
        if N.shape != (n, m):
            raise ValueError(f"X_init_normalized shape {N.shape} does not match expected ({n}, {m})")
        N = N.to(dtype=dtype, device=device)
        print("Using provided initial matching matrix.")
    else:
        # 注意：除以 m，使初始矩阵的行和恰好为 1
        N = torch.ones((n, m), dtype=dtype, device=device) / m

    D = torch.zeros((n, m), dtype=dtype, device=device)
    gamma = gamma0

    # === 【修复 A】矩形匹配的 marginal 质量守恒 ===
    # 注入式匹配（n <= m）：每个源节点必须映射一次（行和=1），
    # 列总质量按 n/m 平均分配，使 sum(src_weight) == sum(trg_weight) == min(n,m)
    if n == m:
        src_weight = torch.ones(n, dtype=dtype, device=device)
        trg_weight = torch.ones(m, dtype=dtype, device=device) 
    elif n < m:
        src_weight = torch.ones(n, dtype=dtype, device=device) 
        trg_weight = torch.ones(m, dtype=dtype, device=device)* (float(n) / float(m))
    else:
        src_weight = torch.ones(n, dtype=dtype, device=device) * (float(m) / float(n))
        trg_weight = torch.ones(m, dtype=dtype, device=device)
    print(f"[Marginal] n={n}, m={m}, sum(src)={src_weight.sum().item():.2f}, "
          f"sum(trg)={trg_weight.sum().item():.2f}")

    # === 【修复 B】lamb 自适应：第一次迭代时按尺度比自动校准 ===
    lamb_adaptive = lamb  # 用户传入的 lamb 视为相对权重（推荐传 1.0）
    lamb_calibrated = False

    for i in range(niter_max):
        N0 = N.clone()

        with autocast(device_type='cuda', enabled=use_amp):
            delta_edges = torch.matmul(torch.matmul(adjacency1, N), adjacency2)

            # 第一次迭代：估算拓扑项与序列项的尺度比，自动校准 lamb
            if not lamb_calibrated and K.numel() > 0 and K.max() > 1e-12:
                topo_scale = delta_edges.abs().max().item()
                seq_scale  = K.max().item()
                if seq_scale > 1e-12 and topo_scale > 1e-12:
                    # 让 lamb*K 的最大值与 delta_edges 的最大值同量级
                    lamb_adaptive = lamb * (topo_scale / seq_scale)
                    print(f"[lamb auto] topo_max={topo_scale:.4e}, seq_max={seq_scale:.4e}, "
                          f"lamb: {lamb} -> {lamb_adaptive:.4e}")
                lamb_calibrated = True

            D[:n, :m] = delta_edges + lamb_adaptive * K

            # ETM — 计算 MROT slack
            D_nm = D[:n, :m]
            u_etm, v_etm, zeta_etm, s = uot_helper(
                D_nm, tau_a=tau, tau_b=tau,
                eps=eps_uot, epoch_iter=uot_iters,
                device=device, dtype=dtype
            )

            etm_alpha = src_weight * torch.exp(
                torch.clamp(-(u_etm + zeta_etm) / tau, min=-500.0, max=500.0)
            )
            etm_beta = trg_weight * torch.exp(
                torch.clamp(-(v_etm - zeta_etm) / tau, min=-500.0, max=500.0)
            )

            # MROT 正则化
            D[:n, :m] = D_nm - lambda_mrot * s

            D_safe = D.clone()
            d_max = D_safe.max()
            if n==m:
                print('n=m')
                if method == 'sinkhorn':
                    D_exp = torch.exp(D - D.max())
                    D = sinkhorn(D_exp, device=device, dtype=dtype)
                elif method == 'asm':
                    D, gamma = adaptive_softassign_torch(D, gamma_ini=1,
                    device=device, dtype=dtype, use_amp=use_amp)
            else:
                print(f'n={n}, m={m}')
                if method == 'sinkhorn':
                    D_exp = torch.exp(D - D.max())
                    D = sinkhorn(D_exp, device=device, dtype=dtype, row_target=etm_alpha, col_target=etm_beta)
                elif method == 'asm':
                    D, gamma = adaptive_softassign_torch(
                        D, gamma_ini=1,
                        device=device, dtype=dtype, use_amp=use_amp,
                        row_target=etm_alpha, col_target=etm_beta)

            # 更新 N
            if adaptive_alpha:
                alpha_op = compute_alpha(N0, delta_edges, D[:n, :m],
                                         adjacency1, adjacency2, K, lamb_adaptive)
                if 0 <= alpha_op < 1:
                    alpha = float(alpha_op)
                    print("Alpha is", alpha_op)
                else:
                    alpha = 1.0

            N = ((1 - alpha) * N + alpha * D[:n, :m]).to(dtype)

        err = torch.linalg.norm(N0 - N, ord="fro") / torch.linalg.norm(N, ord="fro")
        print(f'iter: {i}, err: {err.item():.6f}, gamma: {gamma}')
        if err < tol:
            print("Converged")
            break

    M = hugarian(N.cpu().numpy())   
    runtime = time.perf_counter() - starttime
    return M, N, s, runtime


def sinkhorn(M, num_iters=1000, tol=0.05, device='cuda', dtype=torch.float32, row_target=None, col_target=None):
    if isinstance(M, torch.Tensor):
        M = M.to(dtype=dtype, device=device)
    else:
        M = torch.tensor(np.asarray(M), dtype=dtype, device=device)

    n, m = M.shape

    if row_target is None:
        row_target = torch.ones(n, dtype=dtype, device=device)
    if col_target is None:
        col_target = torch.ones(m, dtype=dtype, device=device)

    u = torch.ones(n, dtype=dtype, device=device)
    v = torch.ones(m, dtype=dtype, device=device)

    for i in range(num_iters):
        u_new = row_target / torch.clamp(torch.matmul(M, v), min=1e-9)
        v_new = col_target / torch.clamp(torch.matmul(M.t(), u_new), min=1e-9)

        if i % 5 == 1:
            res_diff = torch.max(torch.abs(u - u_new)).item()
            if res_diff < tol:
                u, v = u_new, v_new
                break

        u, v = u_new, v_new

    S = torch.outer(u, v) * M
    return S

###############################################################################
# Fusion Moves 集成模块
###############################################################################

def asm_build_config(A1: np.ndarray, A2: np.ndarray, K: np.ndarray,
                     lambda_param: float = 0.5, C_inf: float = 1000000.0):
    cfg = types.SimpleNamespace()
    cfg.A1 = np.array(A1)
    cfg.A2 = np.array(A2)
    cfg.K = np.array(K)
    cfg.lambda_param = lambda_param
    cfg.C_inf = C_inf
    cfg.n1, cfg.n2 = (cfg.A1.shape[0], cfg.A2.shape[0])
    print(f'初始化Fusion模块: {cfg.n1}x{cfg.n2}节点', flush=True)
    cfg.unary_costs = asm_construct_unary_costs(cfg)
    cfg.edge_list_A1 = asm_get_edge_list(cfg, cfg.A1)
    cfg.edge_list_A2 = asm_get_edge_list(cfg, cfg.A2)
    print(f'源图有效边数: {len(cfg.edge_list_A1)}')
    print(f'目标图有效边数: {len(cfg.edge_list_A2)}')
    return cfg

def asm_construct_unary_costs(cfg) -> np.ndarray:
    return -cfg.lambda_param * cfg.K

def asm_get_edge_list(cfg, A: np.ndarray) -> List[Tuple[int, int]]:
    edge_list = []
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                edge_list.append((i, j))
    return edge_list

def asm_compute_pairwise_cost(cfg, u: int, v: int, label_u: int, label_v: int) -> float:
    if u >= cfg.n1 or v >= cfg.n1 or label_u >= cfg.n2 or (label_v >= cfg.n2):
        return 0.0
    return -cfg.A1[u, v] * cfg.A2[label_u, label_v]

def asm_is_feasible(cfg, solution: np.ndarray) -> bool:
    used_labels = set()
    for i, label in enumerate(solution):
        if label >= 0 and label < cfg.n2:
            if label in used_labels:
                return False
            used_labels.add(label)
    return True

def asm_make_feasible(cfg, solution: np.ndarray) -> np.ndarray:
    feasible_solution = solution.copy()
    used_labels = set()
    available_labels = set(range(cfg.n2))
    for i, label in enumerate(feasible_solution):
        if label >= 0 and label < cfg.n2:
            if label not in used_labels:
                used_labels.add(label)
                available_labels.discard(label)
            else:
                feasible_solution[i] = -1
    available_labels = list(available_labels)
    label_idx = 0
    for i in range(len(feasible_solution)):
        if feasible_solution[i] == -1:
            if label_idx < len(available_labels):
                feasible_solution[i] = available_labels[label_idx]
                label_idx += 1
    return feasible_solution

def asm_matrix_to_permutation(cfg, M: np.ndarray) -> np.ndarray:
    M_array = np.array(M)
    permutation = np.full(cfg.n1, -1, dtype=int)
    for i in range(M_array.shape[0]):
        for j in range(M_array.shape[1]):
            if M_array[i, j] == 1:
                permutation[i] = j
                break
    return permutation

# def asm_evaluate_solution(cfg, solution: np.ndarray) -> float:
#     quadratic_term = 0.0
#     for i in range(cfg.n1):
#         for j in range(cfg.n1):
#             if cfg.A1[i, j] != 0:
#                 pi_i = solution[i]
#                 pi_j = solution[j]
#                 if pi_i >= 0 and pi_j >= 0 and (pi_i < cfg.n2) and (pi_j < cfg.n2):
#                     quadratic_term += cfg.A1[i, j] * cfg.A2[pi_i, pi_j]
#     quadratic_term *= 0.5
#     linear_term = 0.0
#     for i in range(cfg.n1):
#         pi_i = solution[i]
#         if pi_i >= 0 and pi_i < cfg.n2:
#             linear_term += cfg.K[i, pi_i]
#     linear_term *= cfg.lambda_param
#     return quadratic_term + linear_term

def asm_evaluate_solution(cfg, solution):
    sol = np.asarray(solution)
    valid = (sol >= 0) & (sol < cfg.n2)
    rows = np.where(valid)[0]
    cols = sol[rows]
    # quadratic
    A1_sub = cfg.A1[np.ix_(rows, rows)]
    A2_sub = cfg.A2[np.ix_(cols, cols)]
    quadratic_term = 0.5 * np.sum(A1_sub * A2_sub)
    # linear
    linear_term = cfg.lambda_param * np.sum(cfg.K[rows, cols])
    return quadratic_term + linear_term


###############################################################################
# 候选解生成 (gradient)
###############################################################################

def graphmatch_gradient(adjacency1, adjacency2, K=0, tol=0.1, alpha=1.0, lamb=1.0, gamma0=1.0,
                         adaptive_alpha=False, niter_max=30, device='cuda', precision='float32',
                         X_init_normalized=None):
    precision = precision.lower()
    use_amp = False

    if precision == 'float64':
        dtype = torch.float64
    elif precision == 'float16':
        dtype = torch.float16
    elif precision == 'tf32':
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif precision == 'amp':
        use_amp = True
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype = torch.float32

    device = torch.device(device)
    starttime = time.perf_counter()

    def to_tensor(x):
        return torch.tensor(x, dtype=dtype, device=device) if not isinstance(x, torch.Tensor) else x.to(dtype=dtype, device=device)

    adjacency1 = to_tensor(adjacency1)
    adjacency2 = to_tensor(adjacency2)
    K = to_tensor(K)

    n, _ = adjacency1.shape
    m, _ = adjacency2.shape
    big_nm = max(n, m)

    if X_init_normalized is not None:
        N = to_tensor(X_init_normalized)
        if N.shape != (n, m):
            raise ValueError(f"X_init_normalized shape {N.shape} does not match expected ({n}, {m})")
        N = N.to(dtype=dtype, device=device)
        print("Using provided initial matching matrix gradient.")
    else:
        N = torch.ones((n, m), dtype=dtype, device=device) / n
        print("Using uniform initial matching matrix.")
    # D = torch.zeros((big_nm, big_nm), dtype=dtype, device=device)
    D = torch.zeros((n, m), dtype=dtype, device=device)

    gamma = gamma0

    N0 = N.clone()

    with autocast(device_type='cuda', enabled=use_amp):
        delta_edges = torch.matmul(torch.matmul(adjacency1, N), adjacency2)
        D[:n, :m] = delta_edges + lamb * K

        D_out_np = hugarian(D.cpu().numpy())
        D_out = torch.from_numpy(np.asarray(D_out_np)).to(device=device, dtype=dtype)
        if adaptive_alpha:
            alpha_op = compute_alpha(N0, delta_edges, D[:n, :m],
                                     adjacency1, adjacency2, K, lamb)
            if 0 <= alpha_op < 1:
                alpha = float(alpha_op)
                print("Alpha is", alpha_op)
            else:
                alpha = 1.0

        N = ((1 - alpha) * N + alpha * D_out[:n, :m]).to(dtype)

    err = torch.norm(N - N0, p='fro') / torch.norm(N0, p='fro')
    print(f'err: {err.item():.6f}, gamma: {gamma}')
    print("Converged")

    M = hugarian(N.cpu().numpy())
    runtime = time.perf_counter() - starttime
    return M, runtime

def asm_generate_candidate_via_asm(cfg, current_solution,
                                   a=0.1,
                                   adaptive_alpha=1,
                                   asm_niter=15, seed=None, device='cuda', precision='float32',
                                   choose: str = 'gradient',
                                   #change=============
                                   source_gw_path=None,
                                   target_gw_path=None,
                                   netal_bin="./Algorithms/Netal/NETAL",
                                   netal_work_dir="./netal_runs",
                                   sana_bin="./Algorithms/SANA/sana2.2",
                                   sana_work_dir="./sana_runs",
                                   #change=============
                                   ):
    if seed is not None:
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.manual_seed(seed)

    n1, n2 = cfg.n1, cfg.n2

    X_current = torch.zeros((n1, n2), dtype=torch.float32, device=device)
    sol = torch.tensor(current_solution, dtype=torch.long, device=device)
    valid = (sol >= 0) & (sol < n2)
    rows = torch.arange(n1, device=device)[valid]
    cols = sol[valid]
    X_current[rows, cols] = 1.0

    X_random = torch.rand((n1, n2), dtype=torch.float32, device=device)
    X_random = sinkhorn(X_random, num_iters=50, device=device)

    X_init = a * X_current + (1 - a) * X_random

    X_init_normalized = sinkhorn(X_init, num_iters=100, tol=0.05)

    if choose == 'gradient':
        candidate_matrix, _ = graphmatch_gradient(
            adjacency1=cfg.A1, adjacency2=cfg.A2, K=cfg.K,
            tol=0.1, lamb=cfg.lambda_param, adaptive_alpha=adaptive_alpha,
            alpha=1.0, gamma0=1.0, niter_max=asm_niter,
            device=device, precision=precision,
            X_init_normalized=X_init_normalized
        )
    elif choose == 'asm':
        candidate_matrix, _, _, _ = graphmatch_ASM_torch(
            adjacency1=cfg.A1, adjacency2=cfg.A2, K=cfg.K,
            tol=0.1, lamb=cfg.lambda_param, adaptive_alpha=adaptive_alpha,
            alpha=1.0, gamma0=1.0, niter_max=asm_niter,
            device=device, precision=precision,
            X_init_normalized=X_init_normalized
        )
     #change=============
    elif choose == 'netal':
        candidate_matrix = run_netal_matching_matrix(
            source_gw_path=source_gw_path,
            target_gw_path=target_gw_path,
            netal_bin=netal_bin,
            work_dir=netal_work_dir,
            # X_init_normalized=X_init_normalized.cpu().numpy()
        )
    elif choose == 'sana':
        candidate_matrix = run_sana_matching_matrix(
            source_gw_path=source_gw_path,
            target_gw_path=target_gw_path,
            sana_bin=sana_bin,
            work_dir=sana_work_dir,       
        )
    #change=============    

    print(f'收敛(GPU加速, {asm_niter}次迭代)')

    candidate_solution = asm_matrix_to_permutation(cfg, candidate_matrix)

    if not asm_is_feasible(cfg, candidate_solution):
        print(f'    警告: ASM生成的候选解不可行,正在修复...')
        candidate_solution = asm_make_feasible(cfg, candidate_solution)

    return candidate_solution

###############################################################################
# QPBO
###############################################################################

def asm_fusion_moves_qpbo(cfg, current_solution, candidate_solution):
    p = np.asarray(current_solution, dtype=int).copy()
    q = np.asarray(candidate_solution, dtype=int).copy()
    n1, n2 = cfg.n1, cfg.n2

    lam = float(cfg.lambda_param)
    A1 = cfg.A1
    A2 = cfg.A2
    K = cfg.K

    sigma = np.empty(n2, dtype=int)
    sigma.fill(-1)
    for i in range(n1):
        sigma[p[i]] = q[i]

    used_right = set(p.tolist())
    visited = np.zeros(n2, dtype=bool)
    cycles = []
    for a in used_right:
        if visited[a]:
            continue
        cur = a
        cyc = []
        while  0 <= cur < n2 and not visited[cur]:
            visited[cur] = True
            cyc.append(cur)
            cur = sigma[cur]    
        cycles.append(cyc)

    m = len(cycles)

    right_to_cycle = np.full(n2, -1, dtype=int)
    for cid, cyc in enumerate(cycles):
        for r in cyc:
            right_to_cycle[r] = cid
    left_cycle = np.array([right_to_cycle[p[i]] for i in range(n1)], dtype=int)

    g = tq.QPBODouble()
    g.add_node(m)

    def pair_contrib(i, j, li, lj):
        return 0.5 * (A1[i, j] * A2[li, lj] + A1[j, i] * A2[lj, li])

    cycle_left = [[] for _ in range(m)]
    for i in range(n1):
        cycle_left[left_cycle[i]].append(i)

    unary0 = np.zeros(m, dtype=float)
    unary1 = np.zeros(m, dtype=float)
    for cid in range(m):
        I = cycle_left[cid]
        for i in I:
            unary0[cid] += -lam * K[i, p[i]]
            unary1[cid] += -lam * K[i, q[i]]
        for idx_a in range(len(I)):
            i = I[idx_a]
            for idx_b in range(idx_a + 1, len(I)):
                j = I[idx_b]
                unary0[cid] += -pair_contrib(i, j, p[i], p[j])
                unary1[cid] += -pair_contrib(i, j, q[i], q[j])

    for cid in range(m):
        g.add_unary_term(cid, float(unary0[cid]), float(unary1[cid]))

    pair_tables = {}
    for i in range(n1):
        ci = left_cycle[i]
        for j in range(i + 1, n1):
            cj = left_cycle[j]
            if ci == cj:
                continue
            if A1[i, j] == 0 and A1[j, i] == 0:
                continue

            c, d = (ci, cj) if ci < cj else (cj, ci)
            if (c, d) not in pair_tables:
                pair_tables[(c, d)] = np.zeros((2, 2), dtype=float)
            tab = pair_tables[(c, d)]

            if ci < cj:
                i_c, j_d = i, j
            else:
                i_c, j_d = j, i

            li0 = p[i_c]; li1 = q[i_c]
            lj0 = p[j_d]; lj1 = q[j_d]

            tab[0, 0] += -pair_contrib(i_c, j_d, li0, lj0)
            tab[0, 1] += -pair_contrib(i_c, j_d, li0, lj1)
            tab[1, 0] += -pair_contrib(i_c, j_d, li1, lj0)
            tab[1, 1] += -pair_contrib(i_c, j_d, li1, lj1)

    for (c, d), tab in pair_tables.items():
        g.add_pairwise_term(int(c), int(d),
                            float(tab[0, 0]), float(tab[0, 1]),
                            float(tab[1, 0]), float(tab[1, 1]))

    g.solve()
    g.improve()

    z = np.array([g.get_label(cid) for cid in range(m)], dtype=int)
    z[z < 0] = 0

    fused = p.copy()
    for i in range(n1):
        if z[left_cycle[i]] == 1:
            fused[i] = q[i]
    return fused

###############################################################################
# 融合函数
###############################################################################

def asm_fusion_moves_optimization(cfg, initial_solution: np.ndarray,
                                  max_iterations: int = 1,
                                  num_candidates: int = 3,
                                  a: float = 0.08,
                                  adaptive_alpha: int = 1,
                                  asm_niter: int = 15,
                                  device: str = 'cuda',
                                  precision: str = 'float32',
                                  choose: str = 'gradient',
                                  #change=============
                                  source_gw_path=None,
                                  target_gw_path=None,
                                  netal_bin="./Algorithms/Netal/NETAL",
                                  netal_work_dir="./netal_runs",
                                  sana_bin="./Algorithms/SANA/sana2.2",
                                  sana_work_dir="./sana_runs",
                                  #change=============
                                  ) -> Tuple[np.ndarray, Dict]:
    print('\n=== 开始Fusion Moves优化 ===', flush=True)
 
    current_solution = initial_solution.copy()
    current_objective = asm_evaluate_solution(cfg, current_solution)

    best_solution = current_solution.copy()
    best_objective = current_objective

    optimization_history = []

    n_nodes = cfg.n1
    initial_accuracy = np.sum(initial_solution == np.arange(n_nodes)) / n_nodes
    all_candidates = []
    all_fused_solutions = []
    fusion_details = []

    print(f'初始解目标值: {current_objective:.4f}', flush=True)
    print(f"初始解可行性: {('✓' if asm_is_feasible(cfg, current_solution) else '✗')}")
    print(f'初始解准确率: {initial_accuracy:.4f}')

    total_candidate_time = 0.0
    total_fusion_time = 0.0

    for iteration in range(max_iterations):
        print(f'\n--- Fusion迭代 {iteration + 1} ---', flush=True)
        improved = False

        iter_cands = []
        iter_fused = []

        print(f'  生成{num_candidates}个双随机候选解:')

        for cand_idx in range(num_candidates):
            print(f'\n  [候选解 {cand_idx + 1}/{num_candidates}]')

            time_start_cand = time.time()
            candidate_solution = asm_generate_candidate_via_asm(
                cfg,
                current_solution=current_solution,
                a=a,
                adaptive_alpha=adaptive_alpha,
                asm_niter=asm_niter,
                seed=None,
                device=device,
                precision=precision,
                choose=choose,
                #change=============
                source_gw_path=source_gw_path,
                target_gw_path=target_gw_path,
                netal_bin=netal_bin,
                netal_work_dir=netal_work_dir,
                sana_bin=sana_bin,
                sana_work_dir=sana_work_dir,
                #change=============
            )
            time_end_cand = time.time()
            dt_cand = time_end_cand - time_start_cand
            total_candidate_time += dt_cand
            print(f'    生成候选解时间: {dt_cand:.2f} 秒')

            if not asm_is_feasible(cfg, candidate_solution):
                    print(f'  候选解 {cand_idx + 1} 不可行，正在修复...')
                    candidate_solution = asm_make_feasible(cfg, candidate_solution)
            iter_cands.append(candidate_solution.copy())

            candidate_objective = asm_evaluate_solution(cfg, candidate_solution)
            candidate_accuracy = np.sum(candidate_solution == np.arange(n_nodes)) / n_nodes
            print(f'  候选解 {cand_idx + 1} 目标值: {candidate_objective:.4f}, 准确率: {candidate_accuracy:.4f}')

            time_start_fusion = time.time()
            fused_solution = asm_fusion_moves_qpbo(cfg, current_solution, candidate_solution)
            time_end_fusion = time.time()
            dt_fusion = time_end_fusion - time_start_fusion
            total_fusion_time += dt_fusion
            print(f'    融合解生成时间: {dt_fusion:.2f} 秒')

            if not asm_is_feasible(cfg, fused_solution):
                 fused_solution = asm_make_feasible(cfg, fused_solution)
            fused_objective = asm_evaluate_solution(cfg, fused_solution)
            fused_accuracy = np.sum(fused_solution == np.arange(n_nodes)) / n_nodes
            print(f'  融合解目标值: {fused_objective:.4f}, 准确率: {fused_accuracy:.4f}')

            iter_fused.append(fused_solution.copy())

            detail = {
                'iteration': iteration,
                'candidate_idx': cand_idx,
                'current_objective': current_objective,
                'current_accuracy': np.sum(current_solution == np.arange(n_nodes)) / n_nodes,
                'candidate_objective': candidate_objective,
                'candidate_accuracy': candidate_accuracy,
                'fused_objective': fused_objective,
                'fused_accuracy': fused_accuracy,
                'current_solution': current_solution.copy(),
                'candidate_solution': candidate_solution.copy(),
                'fused_solution': fused_solution.copy(),
                'total_candidate_time': total_candidate_time,
                'total_fusion_time': total_fusion_time,
            }

            current_objective = asm_evaluate_solution(cfg, current_solution)
            max_objective = max(current_objective, candidate_objective)

            if fused_objective >= max_objective:
                current_solution = fused_solution
                current_objective = fused_objective
                improved = True
                detail['action'] = 'accept_fused'
                print('→ 接受融合解')
            elif fused_objective < max_objective:
                if candidate_objective >= current_objective:
                    current_solution = candidate_solution
                    current_objective = candidate_objective
                    improved = True
                    detail['action'] = 'accept_candidate'
                    print('→ 接受候选解')
                else:
                    detail['action'] = 'keep_current'
                    print('→ 保持当前解')

            fusion_details.append(detail)

            if current_objective > best_objective:
                best_solution = current_solution.copy()
                best_objective = current_objective
                print('-> 更新全局最优解!')

        if len(iter_cands):
            all_candidates.append(np.stack(iter_cands, axis=0))
        if len(iter_fused):
            all_fused_solutions.append(np.stack(iter_fused, axis=0))

        optimization_history.append({
            'iteration': iteration,
            'objective': current_objective,
            'accuracy': np.sum(current_solution == np.arange(n_nodes)) / n_nodes,
            'improved': improved})

    result_info = {
        'initial_objective': asm_evaluate_solution(cfg, initial_solution),
        'final_objective': best_objective,
        'improvement': best_objective - asm_evaluate_solution(cfg, initial_solution),
        'iterations': len(optimization_history),
        'history': optimization_history,
        'candidates_per_iter': all_candidates,
        'candidates_flat': (np.vstack(all_candidates)
                            if len(all_candidates) > 0
                            else np.empty((0, n_nodes), dtype=int)),
        'fused_per_iter': all_fused_solutions,
        'fused_flat': (np.vstack(all_fused_solutions)
                       if len(all_fused_solutions) > 0
                       else np.empty((0, n_nodes), dtype=int)),
        'fusion_details': fusion_details
    }

    print(f'\n=== Fusion Moves完成 ===', flush=True)
    print(f"总候选解生成时间: {total_candidate_time:.2f} 秒")
    print(f"总融合解生成时间: {total_fusion_time:.2f} 秒")
    print(f"总时间(两者之和): {(total_candidate_time + total_fusion_time):.2f} 秒")
    print(f'最终目标值: {best_objective:.4f}')
    print(f"目标值改进: {result_info['improvement']:.4f}")
    return (best_solution, result_info)

###############################################################################
# 评估指标
###############################################################################

def conserved_edges_count(A1, A2, solution):
    if not sp.issparse(A1):
        A1 = sp.csr_matrix(A1)
    if not sp.issparse(A2):
        A2 = sp.csr_matrix(A2)

    n = A1.shape[0]
    rows = np.arange(n)
    cols = solution
    data = np.ones(n)
    P = sp.csr_matrix((data, (rows, cols)), shape=(n, A2.shape[0]))

    aligned_A2 = P @ A2 @ P.T

    common = A1.multiply(aligned_A2)
    Ec = common.sum() / 2.0
    return int(Ec)

def EC(A1, A2, solution):
    if not sp.issparse(A1):
        A1 = sp.csr_matrix(A1)
    E1 = A1.sum() / 2.0
    Ec = conserved_edges_count(A1, A2, solution)
    return float(Ec / E1)

def ICS(A1, A2, solution):
    if not sp.issparse(A1):
        A1 = sp.csr_matrix(A1)
    if not sp.issparse(A2):
        A2 = sp.csr_matrix(A2)

    Ec = conserved_edges_count(A1, A2, solution)

    mapped_nodes = np.asarray(solution)
    A2_sub = A2[mapped_nodes, :][:, mapped_nodes]
    E2_prime = A2_sub.sum() / 2.0

    return float(Ec / E2_prime)

def S3(A1, A2, solution):
    if not sp.issparse(A1):
        A1 = sp.csr_matrix(A1)
    if not sp.issparse(A2):
        A2 = sp.csr_matrix(A2)

    E1 = A1.sum() / 2.0
    Ec = conserved_edges_count(A1, A2, solution)

    mapped_nodes = np.asarray(solution)
    A2_sub = A2[mapped_nodes, :][:, mapped_nodes]
    E2_prime = A2_sub.sum() / 2.0

    return float(Ec / (E1 + E2_prime - Ec))

###############################################################################
# 构建序列相似性矩阵
###############################################################################

def build_K_from_logval(source_graph, target_graph, logval_path):
    source_nodes = list(source_graph.nodes())
    target_nodes = list(target_graph.nodes())
    src2idx = {str(node): i for i, node in enumerate(source_nodes)}
    tgt2idx = {str(node): i for i, node in enumerate(target_nodes)}
    n1, n2 = len(source_nodes), len(target_nodes)
    K = np.zeros((n1, n2), dtype=np.float64)
    with open(logval_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            src_id, tgt_id = parts[0], parts[1]
            score = float(parts[2])
            if src_id in src2idx and tgt_id in tgt2idx:
                i, j = src2idx[src_id], tgt2idx[tgt_id]
                if score > K[i, j]:
                    K[i, j] = score
    print(f"K matrix shape: {K.shape}, non-zero: {np.count_nonzero(K)}")
    return K


def normalize_K(K, top_k=None, cap_quantile=0.99):
    """
    K 鲁棒预处理：截断极端值 + 行 top-k 稀疏化 + 归一化到 [0,1]

    Args:
        K: numpy array, 原始序列相似性矩阵（越大越相似）
        top_k: int or None, 每行只保留 top-k 候选，None 表示不稀疏化
        cap_quantile: float, 截断分位数，超过此分位数的值压平到该分位数
    Returns:
        K_normalized: numpy array, 归一化后的 K, 值域 [0, 1]
    """
    K = np.array(K, dtype=np.float64).copy()

    # 1. 截断极端值（处理 logval 中的 100.0 上限）
    if (K > 0).any():
        cap_value = np.quantile(K[K > 0], cap_quantile)
        K = np.minimum(K, cap_value)

    # 2. 行 top-k 稀疏化（可选）
    if top_k is not None and top_k > 0:
        for i in range(K.shape[0]):
            row = K[i]
            nz = (row > 0).sum()
            if nz > top_k:
                threshold = np.partition(row, -top_k)[-top_k]
                row[row < threshold] = 0
                K[i] = row

    # 3. 归一化到 [0, 1]
    K_max = K.max()
    if K_max > 1e-12:
        K = K / K_max

    print(f"[K normalize] shape={K.shape}, nnz={np.count_nonzero(K)}, "
          f"max={K.max():.4f}, mean(nz)={K[K>0].mean() if (K>0).any() else 0:.4f}")
    return K

###############################################################################
# 批量评测
###############################################################################


name_map = {
    "AT": "AThaliana",
    "CE": "CElegans",
    "DM": "DMelanogaster",
    "MM": "MMusculus",
    "RN": "RNorvegicus",
    "SP": "SPombe",
}

pairs = [
    # ("AT", "DM"),
    ("CE", "AT"),
    # ("CE", "DM"),
    # ("CE", "MM"),
    # ("MM", "AT"),
    # ("MM", "DM"),
    # ("RN", "AT"),
    # ("RN", "CE"),
    # ("RN", "DM"),
    # ("RN", "MM"),
    # ("RN", "SP"),
    # ("SP", "AT"),
    # ("SP", "CE"),
    # ("SP", "DM"),
    # ("SP", "MM"),
]

# =========================================
a_values = [
            # 0.05, 
            # 0.1 , 
            0.5
            ]
lambda_mrot_values = [
                    # 0.1, 
                    # 0.5,
                    #    1, 2
                    3,
                    # 4
                       ]

lambda_param_values = [0]
# =========================================






def evaluate_one_pair(src_abbr, tgt_abbr, a_val, lambda_mrot_val=0.1, lambda_param_val=1.0,
                      #change=============
                    #   candidate_method='gradient'
                      #change=============
                      ):
    src_full = name_map[src_abbr]
    tgt_full = name_map[tgt_abbr]
    pair_name = f"{src_abbr}-{tgt_abbr}"
    print(f"\n{'='*60}", flush=True)
    print(f"  配对: {pair_name}  ({src_full} vs {tgt_full})  a={a_val}  lambda_mrot={lambda_mrot_val}  lambda_param={lambda_param_val}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. 读图 & 邻接矩阵
    source_graph = networkx.read_leda(f"datas/ppi_net/{src_full}.gw")
    target_graph = networkx.read_leda(f"datas/ppi_net/{tgt_full}.gw")
    adj_source = networkx.to_numpy_array(source_graph)
    adj_target = networkx.to_numpy_array(target_graph)
    n1, n2 = adj_source.shape[0], adj_target.shape[0]
    e1 = int(np.sum(adj_source) / 2)
    e2 = int(np.sum(adj_target) / 2)
    print(f"源图节点: {n1}, 边: {e1}")
    print(f"目标图节点: {n2}, 边: {e2}")

    # 2. 构建序列相似性矩阵 K + 鲁棒预处理
    logval_path = f"datas/Sequence_scores/{src_full}-{tgt_full}.logval"
    K_raw = build_K_from_logval(source_graph, target_graph, logval_path)
    K = normalize_K(K_raw, top_k=10, cap_quantile=0.99)

    # 3. ETM+ASM 初始匹配（与 f2.ipynb 参数一致）
    M, N, s, asm_runtime = graphmatch_ASM_torch(
        adj_source, adj_target,
        K=K, alpha=1, lamb=lambda_param_val, adaptive_alpha=0,  # lamb=1.0：内部会自动校准尺度
        niter_max=20, tol=0.1,
        tau=1,
        eps_uot=0.01,
        lambda_mrot=lambda_mrot_val,
        uot_iters=100,
        precision='float32',
        method='asm'
    )
    print(f"ASM runtime: {asm_runtime:.2f}s", flush=True)

    # 4. Fusion Moves 后处理
    # K 已归一化到 [0,1]，但 EC/ICS/S3 只看拓扑，
    # 所以 lambda_param 设小一些，避免 QPBO 偏向序列匹配
    cfg = asm_build_config(
        A1=adj_source, A2=adj_target,
        K=K, lambda_param=lambda_param_val, C_inf=1e7
    )

    initial_solution = asm_matrix_to_permutation(cfg, M)

    #change=============
    src_gw_path = f"datas/ppi_net/{src_full}.gw"
    tgt_gw_path = f"datas/ppi_net/{tgt_full}.gw"
    #change=============
    outer_iter = 0
    current_solution_alt = initial_solution.copy()
    all_fusion_infos = []

    # for outer_iter in range(5):
    final_solution, fusion_info = asm_fusion_moves_optimization(
        cfg,
        initial_solution=current_solution_alt,
        max_iterations=1,
        num_candidates=3,
        a=a_val,
        adaptive_alpha=0,
        asm_niter=5,
        device='cuda',
        precision='float32',
        choose='gradient',
        source_gw_path=src_gw_path,
        target_gw_path=tgt_gw_path,
    )
    all_fusion_infos.append(("gradient", fusion_info))

    # final_solution, fusion_info = asm_fusion_moves_optimization(
    #     cfg,
    #     initial_solution=final_solution1,
    #     max_iterations=1,
    #     num_candidates=1,
    #     a=a_val,
    #     adaptive_alpha=0,
    #     asm_niter=5,
    #     device='cuda',
    #     precision='float32',
    #     choose='sana',
    #     source_gw_path=src_gw_path,
    #     target_gw_path=tgt_gw_path,
    # )
    # all_fusion_infos.append(("sana", fusion_info))

    # current_solution_alt = final_solution.copy()

            
    # 提取 fusion 总时间
    details = fusion_info['fusion_details']
    if len(details) > 0:
        fusion_total_time = (details[-1]['total_candidate_time']
                             + details[-1]['total_fusion_time'])
    else:
        fusion_total_time = 0.0

    # 5. 评估指标
    ec_ini = EC(adj_source, adj_target, initial_solution)
    ec_fin = EC(adj_source, adj_target, final_solution)
    ics_ini = ICS(adj_source, adj_target, initial_solution)
    ics_fin = ICS(adj_source, adj_target, final_solution)
    s3_ini = S3(adj_source, adj_target, initial_solution)
    s3_fin = S3(adj_source, adj_target, final_solution)

    print(f"EC: {ec_ini:.4f} -> {ec_fin:.4f}", flush=True)
    print(f"ICS: {ics_ini:.4f} -> {ics_fin:.4f}", flush=True)
    print(f"S3: {s3_ini:.4f} -> {s3_fin:.4f}", flush=True)

    return {
        'pair': pair_name,
        'a': a_val,
        'lambda_mrot': lambda_mrot_val,
        'lambda_param': lambda_param_val,
        'asm_runtime': round(asm_runtime, 2),
        'fusion_total_time': round(fusion_total_time, 2),
        'EC_ini': round(ec_ini, 6),
        'EC_fin': round(ec_fin, 6),
        'ICS_ini': round(ics_ini, 6),
        'ICS_fin': round(ics_fin, 6),
        'S3_ini': round(s3_ini, 6),
        'S3_fin': round(s3_fin, 6),
    }

###############################################################################
# 主入口
###############################################################################

if __name__ == '__main__':
    results = []
    total_runs = len(pairs) * len(lambda_mrot_values) * len(lambda_param_values) * len(a_values)
    run_idx = 0

    for src_abbr, tgt_abbr in pairs:
        for lm_val in lambda_mrot_values:
            for lp_val in lambda_param_values:
                for a_val in a_values:
                    run_idx += 1
                    try:
                        res = evaluate_one_pair(src_abbr, tgt_abbr, a_val,
                                                lambda_mrot_val=lm_val,
                                                lambda_param_val=lp_val,
                                                )
                        results.append(res)
                    except Exception as e:
                        print(f"\n*** failure: {src_abbr}-{tgt_abbr} a={a_val} lambda_mrot={lm_val} lambda_param={lp_val} — {type(e).__name__}: {e} ***\n", flush=True)
                        continue
                    finally:
                        torch.cuda.empty_cache()
                        gc.collect()
                        print(f"[内存清理完成] ({run_idx}/{total_runs})", flush=True)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("汇总结果:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    results_df.to_excel("difppi_etm_fusion_sana(ec).xlsx", index=False)
    print("\n结果已保存至 difppi_etm_fusion_sana(ec).xlsx")





## ====code modify==== 替换原 name_map / pairs 为新的 source + target 路径列表
# source_graph_path = "networks/networks/synthetic_nets_known_node_mapping/0Krogan_2007_high.gw"

# target_graph_paths = [
#     # "networks/networks/synthetic_nets_known_node_mapping/low_confidence/0Krogan_2007_high+5e.gw",
#     # "networks/networks/synthetic_nets_known_node_mapping/low_confidence/0Krogan_2007_high+10e.gw",
#     # "networks/networks/synthetic_nets_known_node_mapping/low_confidence/0Krogan_2007_high+15e.gw",
#     # "networks/networks/synthetic_nets_known_node_mapping/low_confidence/0Krogan_2007_high+20e.gw",
#     # "networks/networks/synthetic_nets_known_node_mapping/low_confidence/0Krogan_2007_high+25e.gw",
#     "networks/networks/real_world_nets_unknown_node_mapping/yeast.gw",
#     "datas/ppi_net/SCerevisiae.gw",
# ]

# # ====code modify==== 判断 target 是否属于已知 identity mapping 的 synthetic noisy 图
# _known_mapping_dir = "synthetic_nets_known_node_mapping/low_confidence"

# def _has_known_identity_mapping(target_path: str) -> bool:
#     return _known_mapping_dir in target_path.replace("\\", "/")

# a_values = [0.5]
# lambda_mrot_values = [0.5]  # ====code modify==== 新增 lambda_mrot 遍历
# # lambda_mrot_values = [5,10,15]  # ====code modify==== 新增 lambda_mrot 遍历

# # ====code modify==== 将 evaluate_one_pair 改为 evaluate_one_target，适配新数据集 + 增加 NC
# # ====code add==== 基于节点名称构建 ground truth 映射
# def build_name_gt(source_graph, target_graph):
#     """
#     按蛋白质名称匹配，构建 source → target 的 ground truth 排列。
#     返回长度为 n1 的数组 gt，gt[i] = j 表示 source 节点 i 与
#     target 节点 j 名称相同；若目标图中无同名节点则为 -1。
#     """
#     src_nodes = list(source_graph.nodes())
#     tgt_nodes = list(target_graph.nodes())
#     tgt_name2idx = {str(n): j for j, n in enumerate(tgt_nodes)}
#     gt = np.full(len(src_nodes), -1, dtype=int)
#     for i, n in enumerate(src_nodes):
#         j = tgt_name2idx.get(str(n), -1)
#         gt[i] = j
#     matched = int((gt >= 0).sum())
#     print(f"[name-GT] source={len(src_nodes)}, target={len(tgt_nodes)}, "
#           f"matched by name={matched}")
#     return gt


# def NC_name(solution, gt):
#     """
#     按名称 ground truth 计算 NC。
#     分母为源图节点总数（与 SANA 论文一致），
#     分子为 solution[i]==gt[i] 且 gt[i]>=0 的数量。
#     """
#     solution = np.asarray(solution)
#     valid = gt >= 0
#     correct = int(np.sum(solution[valid] == gt[valid]))
#     return float(correct) / len(solution)

# def evaluate_one_target(target_graph_path, a_val, lambda_mrot_val=0.1):
#     pair_name = os.path.basename(target_graph_path)
#     print(f"\n{'='*60}", flush=True)
#     print(f"  target: {pair_name}  a={a_val} lambda_mrot={lambda_mrot_val}", flush=True)
#     print(f"{'='*60}", flush=True)

#     # 1. 读图 & 邻接矩阵
#     source_graph = networkx.read_leda(source_graph_path)
#     target_graph = networkx.read_leda(target_graph_path)
#     adj_source = networkx.to_numpy_array(source_graph)
#     adj_target = networkx.to_numpy_array(target_graph)
#     n1, n2 = adj_source.shape[0], adj_target.shape[0]
#     e1 = int(np.sum(adj_source) / 2)
#     e2 = int(np.sum(adj_target) / 2)
#     print(f"源图节点: {n1}, 边: {e1}")
#     print(f"目标图节点: {n2}, 边: {e2}")

#     # 2. K 矩阵：无序列相似性，直接置零  # ====code modify====
#     K = np.zeros((n1, n2), dtype=np.float64)

#     # 3. ETM+ASM 初始匹配（与 f2.ipynb 参数一致）
#     M, N, s, asm_runtime = graphmatch_ASM_torch(
#         adj_source, adj_target,
#         K=K, alpha=1, lamb=0, adaptive_alpha=0,
#         niter_max=20, tol=0.1,
#         tau=1,
#         eps_uot=0.01,
#         lambda_mrot=lambda_mrot_val,
#         uot_iters=100,
#         precision='float32',
#         method='asm'
#     )
#     print(f"ASM runtime: {asm_runtime:.2f}s", flush=True)

#     # 4. Fusion Moves 后处理
#     cfg = asm_build_config(
#         A1=adj_source, A2=adj_target,
#         K=K, lambda_param=0, C_inf=1e7
#     )

#     initial_solution = asm_matrix_to_permutation(cfg, M)

#     final_solution, fusion_info = asm_fusion_moves_optimization(
#         cfg,
#         initial_solution=initial_solution,
#         max_iterations=1,
#         num_candidates=10,
#         a=a_val,
#         adaptive_alpha=0,
#         asm_niter=5,
#         device='cuda',
#         precision='float32',
#         choose='gradient'
#     )

#     # 提取 fusion 总时间
#     details = fusion_info['fusion_details']
#     if len(details) > 0:
#         fusion_total_time = (details[-1]['total_candidate_time']
#                              + details[-1]['total_fusion_time'])
#     else:
#         fusion_total_time = 0.0

#     # 5. 评估指标
#     ec_ini = EC(adj_source, adj_target, initial_solution)
#     ec_fin = EC(adj_source, adj_target, final_solution)
#     ics_ini = ICS(adj_source, adj_target, initial_solution)
#     ics_fin = ICS(adj_source, adj_target, final_solution)
#     s3_ini = S3(adj_source, adj_target, initial_solution)
#     s3_fin = S3(adj_source, adj_target, final_solution)

#     # ====code modify==== NC 计算
#     if _has_known_identity_mapping(target_graph_path):  
#         nc_ini = float(np.sum(initial_solution == np.arange(n1))) / n1
#         nc_fin = float(np.sum(final_solution   == np.arange(n1))) / n1
#         print(f"NC: {nc_ini:.4f} -> {nc_fin:.4f}", flush=True)
#     else:
#         gt = build_name_gt(source_graph, target_graph)
#         if (gt >= 0).sum() > 0:
#             nc_ini = NC_name(initial_solution, gt)
#             nc_fin = NC_name(final_solution,   gt)
#             print(f"NC (name-match): {nc_ini:.4f} -> {nc_fin:.4f}", flush=True)
#         else:    
#             nc_ini = np.nan
#             nc_fin = np.nan
#             print("NC skipped: no ground-truth node mapping for this target graph.", flush=True)

#     print(f"EC: {ec_ini:.4f} -> {ec_fin:.4f}", flush=True)
#     print(f"ICS: {ics_ini:.4f} -> {ics_fin:.4f}", flush=True)
#     print(f"S3: {s3_ini:.4f} -> {s3_fin:.4f}", flush=True)

#     return {
#         'pair': pair_name,
#         'a': a_val,
#         'lambda_mrot': lambda_mrot_val, 
#         'asm_runtime': round(asm_runtime, 2),
#         'fusion_total_time': round(fusion_total_time, 2),
#         'NC_ini': round(nc_ini, 6) if not np.isnan(nc_ini) else np.nan,  # ====code modify====
#         'NC_fin': round(nc_fin, 6) if not np.isnan(nc_fin) else np.nan,  # ====code modify====
#         'EC_ini': round(ec_ini, 6),
#         'EC_fin': round(ec_fin, 6),
#         'ICS_ini': round(ics_ini, 6),
#         'ICS_fin': round(ics_fin, 6),
#         'S3_ini': round(s3_ini, 6),
#         'S3_fin': round(s3_fin, 6),
#     }

# ###############################################################################
# # 主入口
# ###############################################################################

# # ====code modify==== 主入口改为遍历 target_graph_paths × a_values
# if __name__ == '__main__':
#     results = []
#     total_runs = len(target_graph_paths) * len(lambda_mrot_values) * len(a_values)
#     run_idx = 0

#     for tgt_path in target_graph_paths:
#         for lm_val in lambda_mrot_values:
#             for a_val in a_values:
#                 run_idx += 1
#                 try:
#                     res = evaluate_one_target(tgt_path, a_val, lambda_mrot_val=lm_val)
#                     results.append(res)
#                 except Exception as e:
#                     print(f"\n*** failure: {tgt_path} a={a_val} lambda_mrot={lm_val} — {type(e).__name__}: {e} ***\n", flush=True)
#                     continue
#                 finally:
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     print(f"[内存清理完成] ({run_idx}/{total_runs})", flush=True)

#     results_df = pd.DataFrame(results)
#     print("\n" + "=" * 80)
#     print("汇总结果:")
#     print("=" * 80)
#     print(results_df.to_string(index=False))

#     results_df.to_excel("ppi_etm_noise_lambda345.xlsx", index=False)
#     print("\n结果已保存至 ppi_etm_noise_lambda345.xlsx")
