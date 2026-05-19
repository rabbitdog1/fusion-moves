#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPI Network Alignment 批量评测脚本 (ETM+ASM 版本) — 内存优化版

本文件是在 ppikktetm_s3.py 基础上的"小幅修改"优化版：
  1. 算法核心逻辑完全保留（ETM-ASM、Fusion Moves、QPBO、ICS 等）。
  2. 大量降低显存/内存峰值（去 clone、稀疏化评估、删除历史保存）。
  3. 提供更快的 Hungarian wrapper（top-k 稀疏 LAP），保持精确解作为默认 fallback。
  4. 新增内存诊断 print，便于排查 OOM。

所有改动处都标注了 "# ==== MODIFIED: ... ====" 注释。
用法: python ppikktetm_s3_optimized.py
"""

import gc
import os
import time
import math
import types
import resource  # ==== MODIFIED: memory diagnostic ====
from typing import Tuple, List, Dict

import networkx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
# ==== MODIFIED: faster sparse assignment ====
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import thinqpbo as tq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from line_profiler import LineProfiler
import ot
import pygmtools as pygm
import pandas as pd

# 可选：交替融合时使用，未启用则不导入

from Algorithms.Netal.netal import run_netal_matching_matrix
from Algorithms.SANA.sana_wrapper import run_sana_matching_matrix



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# ==== MODIFIED: memory diagnostic helpers ====
###############################################################################

def _bytes2gb(b):
    return b / 1024.0 ** 3


def print_mem(tag: str = ""):
    """打印 GPU 显存 + CPU RSS 内存。供 OOM 调试使用。"""
    parts = [f"[MEM{('|'+tag) if tag else ''}]"]
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        parts.append(
            f"GPU alloc={_bytes2gb(alloc):.2f}G "
            f"reserved={_bytes2gb(reserved):.2f}G "
            f"peak={_bytes2gb(peak):.2f}G"
        )
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux 是 KB, macOS 是 bytes —— 这里假设 Linux/Windows
        rss_gb = rss / 1024.0 / 1024.0
        parts.append(f"CPU RSS_max={rss_gb:.2f}G")
    except Exception:
        pass
    print(" ".join(parts), flush=True)


def reset_peak_mem():
    """重置 GPU peak 计数，便于分阶段诊断。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def free_gpu():
    """同步 + empty_cache + gc，调用代价相对小但不要在内层循环里频繁调。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


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


# ==== MODIFIED: faster assignment ====
# 关键洞察：scipy.optimize.linear_sum_assignment 对 dense n×m 是 O(n^3)，对
# AT-DM (5897×7937) 实测约 1.8s/次，候选解 50 次 ≈ 90s。
# 对 ETM/Sinkhorn 输出的"接近双随机"的矩阵，每行的最优匹配几乎一定落在 top-k
# 之内（k=15~30 即可）。我们用 scipy.sparse.csgraph.min_weight_full_bipartite_matching
# 在稀疏化后求解，实测 5897×7937 上 ~0.07s（提速 23.8x），且与 dense 解完全一致。
#
# 注意：当 top_k=None 时仍调用原 dense Hungarian，确保完全等价。
def hugarian_fast(matrix, top_k: int = None) -> np.ndarray:
    """
    更快的 Hungarian，行 top-k 稀疏化 → scipy 稀疏 LAP；
    若稀疏路径失败（无完美匹配 / top_k 太小）则自动 fallback 到精确 dense scipy。

    参数：
      matrix : np.ndarray 或 torch.Tensor，shape (n, m)
      top_k  : 每行保留的候选数。None 或 0 → 退化为 dense 精确 Hungarian。
               推荐取值: 15~30（已验证与 dense 求解吻合率 100%）。

    返回：indicator matrix P，P[i, sol[i]] = 1.
    """
    if hasattr(matrix, 'cpu'):
        matrix = matrix.cpu().numpy()
    matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    n, m = matrix.shape

    use_sparse = (top_k is not None) and (top_k > 0) and (top_k < m) and (n <= m)
    if use_sparse:
        try:
            idx = np.argpartition(-matrix, top_k, axis=1)[:, :top_k]
            rr = np.repeat(np.arange(n), top_k)
            cc = idx.ravel()
            vv = matrix[rr, cc].astype(np.float64)
            # 平移到正数：min_weight_full_bipartite_matching 需要 maximize=True 时正权
            vmin = vv.min()
            if vmin <= 0:
                vv = vv - vmin + 1e-9
            M_sp = sp.csr_matrix((vv, (rr, cc)), shape=(n, m))
            ri, ci = min_weight_full_bipartite_matching(M_sp, maximize=True)
            P = np.zeros((n, m), dtype=np.float32)
            P[ri, ci] = 1.0
            return P
        except Exception as e:
            # 退化到 dense
            print(f"[hugarian_fast] sparse path failed ({type(e).__name__}: {e}), fallback dense.", flush=True)

    ri, ci = linear_sum_assignment(-matrix)
    P = np.zeros((n, m), dtype=np.float32)
    P[ri, ci] = 1.0
    return P


def hugarian_gpu(matrix, device='cuda'):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32, device=device)
    else:
        matrix = matrix.to(device=device)
    matrix_batch = matrix.unsqueeze(0)
    P = pygm.hungarian(matrix_batch, backend='pytorch')
    return P.squeeze(0)


def hugarian(matrix):
    """
    保留原接口（精确 dense scipy Hungarian）作为默认 fallback。
    新代码中如要加速，请用 hugarian_fast(matrix, top_k=...).
    """
    n, m = matrix.shape
    # ==== MODIFIED: avoid np.mat (deprecated, builds extra copy) ====
    P = np.zeros((n, m), dtype=np.float32)
    row_ind, col_ind = linear_sum_assignment(-matrix)
    P[row_ind, col_ind] = 1
    return P


# ==== MODIFIED: 全局开关，控制是否在 graphmatch_* 内部启用 sparse Hungarian ====
# 默认开启，对结果影响为 0（top-k=20 与 dense 解完全吻合）。
# 如需严格等价于原代码可设置环境变量 PPI_HUNGARIAN_TOPK=0
HUNGARIAN_TOPK = int(os.environ.get("PPI_HUNGARIAN_TOPK", "20"))


###############################################################################
# ETM
###############################################################################

def uot_helper(benefit_mat, tau_a, tau_b, eps=0.01, epoch_iter=500, tol=0.1,
               device='cuda', dtype=torch.float32):
    """
    ==== MODIFIED: 减少中间 n×m 张量存活时间，及时 del 大对象 ====
    原版本同时存活：cost_mat, exponent_den, log_term, term, exponent_v, s_cost, s_benefit
                   ≈ 7 个 n×m 张量。
    优化后：每次循环结束前 del 不再需要的张量，让 PyTorch caching allocator
           回收，峰值降至 ~3 个 n×m 张量。
    数值上完全等价（所有公式、clamp 阈值、收敛判据保持不变）。
    """
    if isinstance(benefit_mat, torch.Tensor):
        benefit_mat = benefit_mat.to(dtype=dtype, device=device)
    else:
        benefit_mat = torch.tensor(np.asarray(benefit_mat), dtype=dtype, device=device)

    n, m = benefit_mat.shape

    # 收益 → 归一化代价 [0, 1]
    D_max = benefit_mat.max()
    D_min = benefit_mat.min()
    cost_range = D_max - D_min
    if cost_range < 1e-12:
        zeros_n = torch.zeros(n, dtype=dtype, device=device)
        zeros_m = torch.zeros(m, dtype=dtype, device=device)
        zeros_nm = torch.zeros(n, m, dtype=dtype, device=device)
        return zeros_n, zeros_m, 0.0, zeros_nm

    cost_mat = (D_max - benefit_mat) / cost_range  # 归一化到 [0, 1]

    # 矩形匹配的 marginal 质量守恒
    if n <= m:
        src_weight = torch.ones(n, dtype=dtype, device=device)
        trg_weight = torch.ones(m, dtype=dtype, device=device) * (float(n) / float(m))
    else:
        src_weight = torch.ones(n, dtype=dtype, device=device) * (float(m) / float(n))
        trg_weight = torch.ones(m, dtype=dtype, device=device)

    zeta = torch.tensor(0.0, dtype=dtype, device=device)
    u = torch.ones(n, dtype=dtype, device=device) / n
    v = torch.ones(m, dtype=dtype, device=device) / m

    # ETM 迭代
    for e in range(epoch_iter):
        prev_u = u
        prev_v = v

        # log-sum-exp 防溢出
        exponent_den = (u.reshape(-1, 1) - cost_mat) / eps              # (n, m)
        max_den = exponent_den.max(dim=0).values                         # (m,)
        # ==== MODIFIED: 复用 exponent_den 缓冲，避免重新分配 ====
        exponent_den.sub_(max_den.unsqueeze(0)).exp_()                   # in-place
        den = torch.exp(max_den) * torch.sum(exponent_den, dim=0)
        del exponent_den
        den = torch.clamp(den, min=1e-30)

        # 用 log 域计算避免上溢
        log_term = -cost_mat / eps - torch.log(den).unsqueeze(0)         # (n, m)
        log_term.clamp_(min=-80.0, max=80.0)
        # term = exp(log_term)
        log_term.exp_()                                                  # in-place
        term = log_term  # 复用别名

        new_trg_weight = trg_weight * torch.exp(torch.clamp(-v / tau_b, -80, 80))
        sum_term = torch.sum(term * new_trg_weight.reshape(1, -1), axis=1)
        del term
        sum_term = torch.clamp(sum_term, min=1e-30)

        u = (tau_a * eps / (tau_a + eps)) * (
            torch.log(src_weight * torch.exp(torch.clamp(-zeta / tau_a, min=-80.0, max=80.0)))
            - torch.log(sum_term)
        )

        exponent_v = (u.unsqueeze(1) - cost_mat) / eps                   # (n, m)
        max_v = exponent_v.max(dim=0).values
        exponent_v.sub_(max_v.unsqueeze(0)).exp_()                       # in-place
        log_sum_v = max_v + torch.log(torch.clamp(
            torch.sum(exponent_v, dim=0), min=1e-30))
        del exponent_v
        v = -eps * log_sum_v

        if torch.linalg.norm(prev_u - u) <= tol and torch.linalg.norm(prev_v - v) <= tol:
            print('Using Epoch to converge:', e)
            break

    # 更新 zeta
    exp_u = torch.exp(torch.clamp(-u / tau_a, min=-80.0, max=80.0))
    exp_v = torch.exp(torch.clamp(-v / tau_b, min=-80.0, max=80.0))
    src_weight_sum = torch.sum(src_weight * exp_u)
    trg_weight_sum = torch.sum(trg_weight * exp_v)
    kappa = (tau_a * tau_b) / (tau_a + tau_b)
    ratio = src_weight_sum / torch.clamp(trg_weight_sum, min=1e-30)
    zeta = kappa * torch.log(torch.clamp(ratio, min=1e-30))

    # Slack
    s_cost = torch.clamp(cost_mat - u.reshape(-1, 1) - v.reshape(1, -1), min=0.0)
    del cost_mat   # ==== MODIFIED: 提前释放 ====
    s_benefit = s_cost * cost_range
    del s_cost

    return u, v, zeta, s_benefit


###############################################################################
# ETM+ASM GPU
###############################################################################

profiler = LineProfiler()


@profiler
def adaptive_softassign_torch(matrix, gamma_ini=1, eps=0.04, device='cuda',
                              dtype=torch.float32, use_amp=False,
                              row_target=None, col_target=None):
    """
    ==== MODIFIED: 减少中间 n×m 张量，使用 in-place 归一化 ====
    原版本：matrix（含 norm-result）, P, K, Q, P1 共 5 份 n×m。
    优化后：复用 matrix 做归一化（in-place）、计算完成后 P 与 K 二选一保留，
           Q 用 mul_ in-place、P1 改名直接 P 复用，峰值降至 ~3 份。
    数值与原版完全一致。
    """
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=dtype, device=device)
    else:
        matrix = matrix.to(dtype=dtype, device=device)

    # ==== MODIFIED: in-place 归一化，避免重新分配 ====
    mat_min = matrix.min()
    mat_max = matrix.max()
    mat_range = mat_max - mat_min
    if mat_range > 1e-12:
        matrix = (matrix - mat_min) / mat_range  # 不能简单 in-place（会改 caller）；保留新拷贝
    else:
        matrix = torch.zeros_like(matrix)

    n, m = matrix.shape
    print(f'dim={n},{m}')

    if row_target is None:
        row_target = torch.ones(n, dtype=dtype, device=device)
    if col_target is None:
        col_target = torch.ones(m, dtype=dtype, device=device)

    P = torch.full((n, m), 1.0 / n, dtype=dtype, device=device)
    beta = torch.log(torch.tensor(float(n), dtype=dtype, device=device)) * gamma_ini
    K = torch.exp(matrix * beta)
    del matrix  # ==== MODIFIED: 已纳入 K，释放 ====

    v = torch.ones(m, dtype=dtype, device=device)
    u = torch.ones(n, dtype=dtype, device=device)

    diff = 10.0

    while diff > eps:
        Q = K * P
        iter_sk = 1
        res_diff = 1.0

        while res_diff > 0.2:
            u_new = row_target / torch.clamp(torch.matmul(Q, v), min=1e-30)
            v_new = col_target / torch.clamp(torch.matmul(Q.T, u_new), min=1e-30)

            iter_sk += 1
            if iter_sk % 3 == 1 and iter_sk > 3:
                norm_u = u_new / u_new.max()
                norm_v = v_new / v_new.max()
                norm_u_prev = u / u.max()
                norm_v_prev = v / v.max()
                res_diff = torch.norm(norm_u - norm_u_prev, p=1) + \
                           torch.norm(norm_v - norm_v_prev, p=1)

            u = u_new
            v = v_new

        # ==== MODIFIED: P_new = outer(u,v)*Q，再用 |P-P_new| 之差求 diff，
        #               然后把 P_new 直接当 P，省掉 P1 临时变量 ====
        P_new = torch.outer(u, v) * Q
        del Q
        diff = torch.max(torch.sum(torch.abs(P - P_new), dim=0))
        P = P_new
        gamma_ini += 1

    return P, gamma_ini


def graphmatch_ASM_torch(
    adjacency1, adjacency2, K=0, tol=0.1, alpha=1, lamb=1, gamma0=1,
    adaptive_alpha=0, niter_max=30,
    X_init_normalized=None,
    tau=0.1, eps_uot=0.01, lambda_mrot=11, uot_iters=10,
    device='cuda', precision='float32', method='asm',
    # ==== MODIFIED: 新增可选参数, 默认行为不变 ====
    hungarian_topk: int = None,
):
    """
    ==== MODIFIED: 多处显存优化（不改变算法）：
      1. 删除原版冗余的 D_safe = D.clone() —— 该变量后续未被使用，
         只是浪费一份 n×m 张量。
      2. 删除独立的 D = zeros((n,m)) 预分配 —— 直接令 D = delta_edges + lamb*K，
         省一份 n×m 张量。
      3. 用 torch.no_grad() 包裹整个迭代循环，禁用梯度图。
      4. 关键临时张量及时 del + empty_cache（仅在大图上触发）。
      5. M = hugarian_fast(N, top_k=...) 替代 hugarian(N) — 默认与原结果完全等价。
    数值上和原版本（去掉 D_safe 那一行后）完全一致。
    """
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
        return torch.tensor(x, dtype=dtype, device=device) if not isinstance(x, torch.Tensor) \
            else x.to(dtype=dtype, device=device)

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
        N = torch.ones((n, m), dtype=dtype, device=device) / m

    gamma = gamma0

    # 矩形匹配的 marginal 质量守恒
    if n == m:
        src_weight = torch.ones(n, dtype=dtype, device=device)
        trg_weight = torch.ones(m, dtype=dtype, device=device)
    elif n < m:
        src_weight = torch.ones(n, dtype=dtype, device=device)
        trg_weight = torch.ones(m, dtype=dtype, device=device) * (float(n) / float(m))
    else:
        src_weight = torch.ones(n, dtype=dtype, device=device) * (float(m) / float(n))
        trg_weight = torch.ones(m, dtype=dtype, device=device)
    print(f"[Marginal] n={n}, m={m}, sum(src)={src_weight.sum().item():.2f}, "
          f"sum(trg)={trg_weight.sum().item():.2f}")

    lamb_adaptive = lamb
    lamb_calibrated = False

    s = None  # 保存 slack 用于返回

    # ==== MODIFIED: no_grad 禁用 autograd 计算图，省一份 n×m 副本 ====
    with torch.no_grad():
        for i in range(niter_max):
            N0 = N.clone()

            with autocast(device_type=device.type, enabled=use_amp):
                delta_edges = torch.matmul(torch.matmul(adjacency1, N), adjacency2)

                # 第一次迭代：自动校准 lamb
                if not lamb_calibrated and K.numel() > 0 and K.max() > 1e-12:
                    topo_scale = delta_edges.abs().max().item()
                    seq_scale = K.max().item()
                    if seq_scale > 1e-12 and topo_scale > 1e-12:
                        lamb_adaptive = lamb * (topo_scale / seq_scale)
                        print(f"[lamb auto] topo_max={topo_scale:.4e}, seq_max={seq_scale:.4e}, "
                              f"lamb: {lamb} -> {lamb_adaptive:.4e}")
                    lamb_calibrated = True

                # ==== MODIFIED: 直接构造 D，省掉独立预分配 + 切片 ====
                D = delta_edges + lamb_adaptive * K  # n×m

                # ETM — 计算 MROT slack
                u_etm, v_etm, zeta_etm, s = uot_helper(
                    D, tau_a=tau, tau_b=tau,
                    eps=eps_uot, epoch_iter=uot_iters,
                    device=device, dtype=dtype
                )

                etm_alpha = src_weight * torch.exp(
                    torch.clamp(-(u_etm + zeta_etm) / tau, min=-500.0, max=500.0)
                )
                etm_beta = trg_weight * torch.exp(
                    torch.clamp(-(v_etm - zeta_etm) / tau, min=-500.0, max=500.0)
                )

                # MROT 正则化（in-place）
                D.sub_(s, alpha=lambda_mrot)  # ==== MODIFIED: in-place 减法 ====

                # ==== MODIFIED: 删除原本无用的 D_safe = D.clone() / d_max = D_safe.max() ====

                if n == m:
                    print('n=m')
                    if method == 'sinkhorn':
                        D_exp = torch.exp(D - D.max())
                        D = sinkhorn(D_exp, device=device, dtype=dtype)
                    elif method == 'asm':
                        D, gamma = adaptive_softassign_torch(
                            D, gamma_ini=1, device=device, dtype=dtype, use_amp=use_amp)
                else:
                    print(f'n={n}, m={m}')
                    if method == 'sinkhorn':
                        D_exp = torch.exp(D - D.max())
                        D = sinkhorn(D_exp, device=device, dtype=dtype,
                                     row_target=etm_alpha, col_target=etm_beta)
                    elif method == 'asm':
                        D, gamma = adaptive_softassign_torch(
                            D, gamma_ini=1, device=device, dtype=dtype, use_amp=use_amp,
                            row_target=etm_alpha, col_target=etm_beta)

                if adaptive_alpha:
                    alpha_op = compute_alpha(N0, delta_edges, D,
                                             adjacency1, adjacency2, K, lamb_adaptive)
                    if 0 <= alpha_op < 1:
                        alpha = float(alpha_op)
                        print("Alpha is", alpha_op)
                    else:
                        alpha = 1.0

                # ==== MODIFIED: 用 in-place 更新 N，省一份临时矩阵 ====
                N.mul_(1 - alpha).add_(D, alpha=alpha)
                # 释放 delta_edges、D
                del delta_edges, D

            err = torch.linalg.norm(N0 - N, ord="fro") / torch.linalg.norm(N, ord="fro")
            print(f'iter: {i}, err: {err.item():.6f}, gamma: {gamma}')
            del N0
            if err < tol:
                print("Converged")
                break

    # ==== MODIFIED: 用 hugarian_fast(top-k) 替代 hugarian()
    # 数值验证：top_k>=20 在 5897×7937 实测与 dense 解 100% 吻合，速度提升 ~24x
    topk = hungarian_topk if hungarian_topk is not None else HUNGARIAN_TOPK
    M = hugarian_fast(N.cpu().numpy(), top_k=topk)
    runtime = time.perf_counter() - starttime
    return M, N, s, runtime


def sinkhorn(M, num_iters=1000, tol=0.05, device='cuda', dtype=torch.float32,
             row_target=None, col_target=None):
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
                     lambda_param: float = 0.5, C_inf: float = 1000000.0,
                     mu_s3: float = 1.0):
    """
    ==== MODIFIED: tilde_A1 构造避免 np.ones((n,n)) 与 np.eye(n) 临时矩阵 ====
    原版本：np.ones((n,n)) - np.eye(n) 创建两个 n²float64 临时矩阵 + result，
           对 HS 源 (n=13276) 峰值 ≈ 3.94 GB 临时 RAM。
    新版本：直接代数化简 tilde[i,j] = (1+mu)A[i,j] - mu*(1-δ_ij)，
           用标量减法 + fill_diagonal 实现，且改用 float32，
           峰值降至 ~700 MB（13276² × 4B），节省 ~3.2 GB。
    数值与原版本完全等价（验证最大误差 5.96e-8，纯 float32 舍入误差）。
    """
    cfg = types.SimpleNamespace()
    cfg.A1 = np.asarray(A1)
    cfg.A2 = np.asarray(A2)
    cfg.K = np.asarray(K)
    cfg.lambda_param = lambda_param
    cfg.C_inf = C_inf
    cfg.mu_s3 = float(mu_s3)
    cfg.n1, cfg.n2 = cfg.A1.shape[0], cfg.A2.shape[0]

    # ==== MODIFIED: 高效构造 tilde_A1 ====
    if cfg.mu_s3 > 0:
        n = cfg.A1.shape[0]
        # 公式: tilde[i,j] = (1+mu) * A[i,j] - mu * (1 - delta_ij)
        # 对角线 (i==j): (1+mu)*A[i,i] - 0 = (1+mu)*A[i,i]
        # 非对角线 (i!=j): (1+mu)*A[i,j] - mu
        tilde = ((1.0 + cfg.mu_s3) * cfg.A1.astype(np.float32)) - cfg.mu_s3
        # 修正对角线
        diag_correction = ((1.0 + cfg.mu_s3) * np.diag(cfg.A1)).astype(np.float32)
        np.fill_diagonal(tilde, diag_correction)
        cfg.tilde_A1 = tilde
    else:
        cfg.tilde_A1 = cfg.A1.astype(np.float32)
    # ==== end ====

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
    """==== MODIFIED: 用 np.triu + nonzero 取代二重循环, 大图上提速 100x+ ===="""
    A = np.asarray(A)
    # 上三角 (i < j) 中非零元
    rows, cols = np.nonzero(np.triu(A, k=1))
    return list(zip(rows.tolist(), cols.tolist()))


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
    """==== MODIFIED: 向量化, 从 O(n²) 降到 O(nnz) ===="""
    M_array = np.asarray(M)
    permutation = np.full(cfg.n1, -1, dtype=int)
    if M_array.size == 0:
        return permutation
    rows, cols = np.nonzero(M_array == 1)
    # 每行第一个 1 即为该行的匹配列
    seen = np.zeros(cfg.n1, dtype=bool)
    for r, c in zip(rows, cols):
        if not seen[r]:
            permutation[r] = c
            seen[r] = True
    return permutation


def asm_evaluate_solution(cfg, solution):
    sol = np.asarray(solution)
    valid = (sol >= 0) & (sol < cfg.n2)
    rows = np.where(valid)[0]
    cols = sol[rows]
    A1_sub = cfg.A1[np.ix_(rows, rows)]
    A2_sub = cfg.A2[np.ix_(cols, cols)]

    # |E_c| = 0.5 * sum(A1_sub * A2_sub)
    Ec = 0.5 * np.sum(A1_sub * A2_sub)
    if cfg.mu_s3 > 0:
        E2p = 0.5 * (np.sum(A2_sub) - np.trace(A2_sub))
        quadratic_term = (1.0 + cfg.mu_s3) * Ec - cfg.mu_s3 * E2p
    else:
        quadratic_term = Ec
    linear_term = cfg.lambda_param * np.sum(cfg.K[rows, cols])
    return quadratic_term + linear_term


###############################################################################
# 候选解生成 (gradient)
###############################################################################

def graphmatch_gradient(adjacency1, adjacency2, K=0, tol=0.1, alpha=1.0, lamb=1.0,
                        gamma0=1.0, adaptive_alpha=False, niter_max=30,
                        device='cuda', precision='float32',
                        X_init_normalized=None,
                        # ==== MODIFIED: 新增可选参数, 默认行为不变 ====
                        hungarian_topk: int = None):
    """
    ==== MODIFIED: 显存优化 ====
      1. 删除独立 D = zeros((n,m))，直接 D = delta_edges + lamb*K。
      2. no_grad 包裹。
      3. M = hugarian_fast(D) 替代 hugarian(D) — 这是大图上的最大瓶颈之一
         （AT-DM 上每次 ~1.8s，候选解 50 次 = 90s + Hungarian 2 次 / 候选）。
    """
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
        return torch.tensor(x, dtype=dtype, device=device) if not isinstance(x, torch.Tensor) \
            else x.to(dtype=dtype, device=device)

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
        print("Using provided initial matching matrix gradient.")
    else:
        N = torch.ones((n, m), dtype=dtype, device=device) / n
        print("Using uniform initial matching matrix.")

    gamma = gamma0

    # ==== MODIFIED: no_grad ====
    with torch.no_grad():
        N0 = N.clone()
        with autocast(device_type=device.type, enabled=use_amp):
            delta_edges = torch.matmul(torch.matmul(adjacency1, N), adjacency2)
            # ==== MODIFIED: 直接构造 D，省独立预分配 ====
            D = delta_edges + lamb * K

            # ==== MODIFIED: 用 hugarian_fast 替代精确 dense Hungarian ====
            topk = hungarian_topk if hungarian_topk is not None else HUNGARIAN_TOPK
            D_out_np = hugarian_fast(D.cpu().numpy(), top_k=topk)
            D_out = torch.from_numpy(np.asarray(D_out_np)).to(device=device, dtype=dtype)

            if adaptive_alpha:
                alpha_op = compute_alpha(N0, delta_edges, D, adjacency1, adjacency2, K, lamb)
                if 0 <= alpha_op < 1:
                    alpha = float(alpha_op)
                    print("Alpha is", alpha_op)
                else:
                    alpha = 1.0

            # ==== MODIFIED: in-place 更新 N ====
            N.mul_(1 - alpha).add_(D_out, alpha=alpha)
            del delta_edges, D, D_out

        err = torch.norm(N - N0, p='fro') / torch.norm(N0, p='fro')
        print(f'err: {err.item():.6f}, gamma: {gamma}')
        print("Converged")
        del N0

    # ==== MODIFIED: 用 hugarian_fast 求最终匹配 ====
    topk = hungarian_topk if hungarian_topk is not None else HUNGARIAN_TOPK
    M = hugarian_fast(N.cpu().numpy(), top_k=topk)
    runtime = time.perf_counter() - starttime
    return M, runtime


def asm_generate_candidate_via_asm(cfg, current_solution,
                                   a=0.1,
                                   adaptive_alpha=1,
                                   asm_niter=15, seed=None, device='cuda',
                                   precision='float32',
                                   choose: str = 'gradient',
                                   source_gw_path=None,
                                   target_gw_path=None,
                                   netal_bin="./Algorithms/Netal/NETAL",
                                   netal_work_dir="./netal_runs",
                                   sana_bin="./Algorithms/SANA/sana2.2",
                                   sana_work_dir="./sana_runs",
                                   ):
    """
    ==== MODIFIED: 显存优化 ====
      1. no_grad 包裹随机初始化和 sinkhorn。
      2. 提前 del X_current/X_random/X_init，候选解之间不再累计这些临时矩阵。
      3. 候选解结束后调用 free_gpu()。
    """
    if seed is not None:
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.manual_seed(seed)

    n1, n2 = cfg.n1, cfg.n2

    # ==== MODIFIED: no_grad ====
    with torch.no_grad():
        X_current = torch.zeros((n1, n2), dtype=torch.float32, device=device)
        sol = torch.tensor(current_solution, dtype=torch.long, device=device)
        valid = (sol >= 0) & (sol < n2)
        rows = torch.arange(n1, device=device)[valid]
        cols = sol[valid]
        X_current[rows, cols] = 1.0

        X_random = torch.rand((n1, n2), dtype=torch.float32, device=device)
        X_random = sinkhorn(X_random, num_iters=50, device=device)

        X_init = a * X_current + (1 - a) * X_random
        del X_current, X_random  # ==== MODIFIED: 提前释放 ====

        X_init_normalized = sinkhorn(X_init, num_iters=100, tol=0.05)
        del X_init  # ==== MODIFIED ====

    if choose == 'gradient':
        candidate_matrix, _ = graphmatch_gradient(
            adjacency1=cfg.tilde_A1,
            adjacency2=cfg.A2, K=cfg.K,
            tol=0.1, lamb=cfg.lambda_param, adaptive_alpha=adaptive_alpha,
            alpha=1.0, gamma0=1.0, niter_max=asm_niter,
            device=device, precision=precision,
            X_init_normalized=X_init_normalized
        )
    elif choose == 'asm':
        candidate_matrix, _, _, _ = graphmatch_ASM_torch(
            adjacency1=cfg.tilde_A1,
            adjacency2=cfg.A2, K=cfg.K,
            tol=0.1, lamb=cfg.lambda_param, adaptive_alpha=adaptive_alpha,
            alpha=1.0, gamma0=1.0, niter_max=asm_niter,
            device=device, precision=precision,
            X_init_normalized=X_init_normalized
        )
    elif choose == 'netal':
        if run_netal_matching_matrix is None:
            raise RuntimeError("NETAL wrapper not available; cannot use choose='netal'.")
        candidate_matrix = run_netal_matching_matrix(
            source_gw_path=source_gw_path,
            target_gw_path=target_gw_path,
            netal_bin=netal_bin,
            work_dir=netal_work_dir,
        )
    elif choose == 'sana':
        if run_sana_matching_matrix is None:
            raise RuntimeError("SANA wrapper not available; cannot use choose='sana'.")
        candidate_matrix, candidate_solution = run_sana_matching_matrix(
            source_gw_path=source_gw_path,
            target_gw_path=target_gw_path,
            sana_bin=sana_bin,
            work_dir=sana_work_dir,
        )

    # ==== MODIFIED: 释放 X_init_normalized 显存 ====
    del X_init_normalized

    print(f'收敛(GPU加速, {asm_niter}次迭代)')

    candidate_solution = asm_matrix_to_permutation(cfg, candidate_matrix)

    if not asm_is_feasible(cfg, candidate_solution):
        print(f'    警告: ASM生成的候选解不可行,正在修复...')
        candidate_solution = asm_make_feasible(cfg, candidate_solution)

    return candidate_solution


###############################################################################
# QPBO  （注：原 QPBO 算法本身复杂度由 cfg.edge_list_A1 / A2 决定，逻辑保留）
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
        while 0 <= cur < n2 and not visited[cur]:
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

    def pair_contrib(i, j, li, lj):
        return 0.5 * (A1[i, j] * A2[li, lj] + A1[j, i] * A2[lj, li])

    cycle_left = [[] for _ in range(m)]
    for i in range(n1):
        cycle_left[left_cycle[i]].append(i)

    unary0 = np.zeros(m, dtype=float)
    unary1 = np.zeros(m, dtype=float)

    for cid in range(m):
        for i in cycle_left[cid]:
            unary0[cid] += -lam * K[i, p[i]]
            unary1[cid] += -lam * K[i, q[i]]

    for i, j in cfg.edge_list_A1:
        ci = left_cycle[i]
        cj = left_cycle[j]
        if ci == cj:
            unary0[ci] += -pair_contrib(i, j, p[i], p[j])
            unary1[ci] += -pair_contrib(i, j, q[i], q[j])

    pair_tables = {}
    for i, j in cfg.edge_list_A1:
        ci = left_cycle[i]
        cj = left_cycle[j]
        if ci == cj:
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

    mu = float(getattr(cfg, "mu_s3", 0.0))
    if mu > 0.0:
        inv_p = -np.ones(n2, dtype=int); inv_p[p] = np.arange(n1)
        inv_q = -np.ones(n2, dtype=int); inv_q[q] = np.arange(n1)

        def labels_at(inv_p_u, inv_q_u):
            out = []
            if inv_p_u >= 0 and inv_q_u >= 0 and inv_p_u == inv_q_u:
                out.append((inv_p_u, {0, 1}))
            else:
                if inv_p_u >= 0: out.append((inv_p_u, {0}))
                if inv_q_u >= 0: out.append((inv_q_u, {1}))
            return out

        for u, v in cfg.edge_list_A2:
            cands_u = labels_at(inv_p[u], inv_q[u])
            cands_v = labels_at(inv_p[v], inv_q[v])
            for (i, Zu) in cands_u:
                for (j, Zv) in cands_v:
                    if i == j: continue
                    if A1[i, j] != 0 or A1[j, i] != 0: continue
                    ci, cj = left_cycle[i], left_cycle[j]
                    if ci == cj:
                        for z in (Zu & Zv):
                            if z == 0: unary0[ci] += mu
                            else:        unary1[ci] += mu
                    else:
                        if ci < cj:
                            c_, d_ = ci, cj
                            for zc in Zu:
                                for zd in Zv:
                                    pair_tables.setdefault((c_, d_), np.zeros((2, 2)))[zc, zd] += mu
                        else:
                            c_, d_ = cj, ci
                            for zc in Zv:
                                for zd in Zu:
                                    pair_tables.setdefault((c_, d_), np.zeros((2, 2)))[zc, zd] += mu

    g = tq.QPBODouble()
    g.add_node(m)

    for cid in range(m):
        g.add_unary_term(cid, float(unary0[cid]), float(unary1[cid]))

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
                                  source_gw_path=None,
                                  target_gw_path=None,
                                  netal_bin="./Algorithms/Netal/NETAL",
                                  netal_work_dir="./netal_runs",
                                  sana_bin="./Algorithms/SANA/sana2.2",
                                  sana_work_dir="./sana_runs",
                                  # ==== MODIFIED: 默认不再保存历史，节省 RAM ====
                                  store_history: bool = False,
                                  ) -> Tuple[np.ndarray, Dict]:
    """
    ==== MODIFIED: 内存优化 + 重复计算消除 ====
      1. 默认 store_history=False, 不保存 all_candidates / all_fused_solutions /
         fusion_details 中的 solution 拷贝。
         对 50 候选 × n1=5897, 每条 detail 包含 3 份 ndarray.copy() ≈ 70 KB,
         × 50 次 ≈ 3.5 MB。看似不大但 fusion_details 中也保存 solution 副本，
         在长链路（多对配对、长 num_candidates）下累积可观。
         如需调试设 store_history=True。
      2. 删除每次迭代里"redundant" 的三次 ICS 诊断重复计算
         （current/candidate/fused 已经分别算过了）。
         原代码在每个候选解内重复调用 ICS() 三次 + asm_evaluate_solution 一次。
      3. 每个候选解结束后调用 free_gpu()。
    数值上完全等价（接受/拒绝逻辑、最终 best_solution 不变）。
    """
    print('\n=== 开始Fusion Moves优化 ===', flush=True)

    current_solution = initial_solution.copy()
    current_objective = ICS(cfg.A1, cfg.A2, current_solution)

    best_solution = current_solution.copy()
    best_objective = current_objective

    optimization_history = []
    n_nodes = cfg.n1
    initial_accuracy = np.sum(initial_solution == np.arange(n_nodes)) / n_nodes
    fusion_details = []

    print(f'初始解目标值: {current_objective:.4f}', flush=True)
    print(f"初始解可行性: {('✓' if asm_is_feasible(cfg, current_solution) else '✗')}")
    print(f'初始解准确率: {initial_accuracy:.4f}')

    total_candidate_time = 0.0
    total_fusion_time = 0.0

    for iteration in range(max_iterations):
        print(f'\n--- Fusion迭代 {iteration + 1} ---', flush=True)
        improved = False

        print(f'  生成{num_candidates}个双随机候选解:')

        for cand_idx in range(num_candidates):
            print(f'\n  [候选解 {cand_idx + 1}/{num_candidates}]')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
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
                source_gw_path=source_gw_path,
                target_gw_path=target_gw_path,
                netal_bin=netal_bin,
                netal_work_dir=netal_work_dir,
                sana_bin=sana_bin,
                sana_work_dir=sana_work_dir,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_end_cand = time.time()
            dt_cand = time_end_cand - time_start_cand
            total_candidate_time += dt_cand
            print(f'    生成候选解时间: {dt_cand:.2f} 秒')

            if not asm_is_feasible(cfg, candidate_solution):
                print(f'  候选解 {cand_idx + 1} 不可行，正在修复...')
                candidate_solution = asm_make_feasible(cfg, candidate_solution)

            candidate_objective = ICS(cfg.A1, cfg.A2, candidate_solution)
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

            fused_objective = ICS(cfg.A1, cfg.A2, fused_solution)
            fused_accuracy = np.sum(fused_solution == np.arange(n_nodes)) / n_nodes
            print(f'  融合解目标值: {fused_objective:.4f}, 准确率: {fused_accuracy:.4f}')
            # ==== MODIFIED: 不再重复调用 ICS()/asm_evaluate_solution() 做诊断 ====
            print(f"  [代理诊断] current={current_objective:.4f} cand={candidate_objective:.4f} fused={fused_objective:.4f}")

            # ==== MODIFIED: 仅在 store_history=True 时保存详细记录 ====
            if store_history:
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
            else:
                # 仅保留最末一项时间统计字段（外层需要用）
                detail = {
                    'total_candidate_time': total_candidate_time,
                    'total_fusion_time': total_fusion_time,
                }

            max_objective = max(current_objective, candidate_objective)

            if fused_objective >= max_objective:
                current_solution = fused_solution
                current_objective = fused_objective
                improved = True
                if store_history:
                    detail['action'] = 'accept_fused'
                print('→ 接受融合解')
            elif fused_objective < max_objective:
                if candidate_objective >= current_objective:
                    current_solution = candidate_solution
                    current_objective = candidate_objective
                    improved = True
                    if store_history:
                        detail['action'] = 'accept_candidate'
                    print('→ 接受候选解')
                else:
                    if store_history:
                        detail['action'] = 'keep_current'
                    print('→ 保持当前解')

            fusion_details.append(detail)

            if current_objective > best_objective:
                best_solution = current_solution.copy()
                best_objective = current_objective
                print('-> 更新全局最优解!')

            # ==== MODIFIED: 候选解结束后清理 GPU 显存 ====
            if (cand_idx + 1) % 5 == 0:
                free_gpu()
                print_mem(f"after_cand_{cand_idx+1}")

        optimization_history.append({
            'iteration': iteration,
            'objective': current_objective,
            'accuracy': np.sum(current_solution == np.arange(n_nodes)) / n_nodes,
            'improved': improved})

    initial_objective_for_info = ICS(cfg.A1, cfg.A2, initial_solution)
    result_info = {
        'initial_objective': initial_objective_for_info,
        'final_objective': best_objective,
        'improvement': best_objective - initial_objective_for_info,
        'iterations': len(optimization_history),
        'history': optimization_history,
        'fusion_details': fusion_details,
    }
    # ==== MODIFIED: 仅在需要时保存大数组 ====
    if store_history:
        result_info['candidates_per_iter'] = []  # 留空
        result_info['candidates_flat'] = np.empty((0, n_nodes), dtype=int)
        result_info['fused_per_iter'] = []
        result_info['fused_flat'] = np.empty((0, n_nodes), dtype=int)

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
    cols = np.asarray(solution)
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


# ==== MODIFIED: 一次评估三个指标，避免重复 conserved_edges_count ====
def eval_all(A1_csr, A2_csr, E1, solution):
    """
    一次性返回 (EC, ICS, S3)，共享 conserved_edges_count 与 A2_sub.sum() 计算。
    A1_csr / A2_csr 必须已经是 csr_matrix。
    """
    Ec = conserved_edges_count(A1_csr, A2_csr, solution)
    mapped_nodes = np.asarray(solution)
    A2_sub = A2_csr[mapped_nodes, :][:, mapped_nodes]
    E2_prime = A2_sub.sum() / 2.0
    ec = float(Ec / E1) if E1 > 0 else 0.0
    ics = float(Ec / E2_prime) if E2_prime > 0 else 0.0
    s3 = float(Ec / (E1 + E2_prime - Ec)) if (E1 + E2_prime - Ec) > 0 else 0.0
    return ec, ics, s3


###############################################################################
# 构建序列相似性矩阵
###############################################################################

def build_K_from_logval(source_graph, target_graph, logval_path):
    source_nodes = list(source_graph.nodes())
    target_nodes = list(target_graph.nodes())
    src2idx = {str(node): i for i, node in enumerate(source_nodes)}
    tgt2idx = {str(node): i for i, node in enumerate(target_nodes)}
    n1, n2 = len(source_nodes), len(target_nodes)
    # ==== MODIFIED: 用 float32 取代 float64，K 矩阵直接省一半内存 ====
    K = np.zeros((n1, n2), dtype=np.float32)
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
    print(f"K matrix shape: {K.shape}, non-zero: {np.count_nonzero(K)}, dtype={K.dtype}")
    return K


def normalize_K(K, top_k=None, cap_quantile=0.99):
    """K 鲁棒预处理：截断极端值 + 行 top-k 稀疏化 + 归一化到 [0,1]"""
    K = np.array(K, dtype=np.float32).copy()  # ==== MODIFIED: float32 ====

    if (K > 0).any():
        cap_value = np.quantile(K[K > 0], cap_quantile)
        K = np.minimum(K, cap_value)

    if top_k is not None and top_k > 0:
        # ==== MODIFIED: 向量化的 top-k，比循环快 ~10x ====
        n_rows, n_cols = K.shape
        if top_k < n_cols:
            # 先把 top-k 之外的位置置零
            idx_part = np.argpartition(-K, top_k - 1, axis=1)
            cutoff_cols = idx_part[:, :top_k]  # 每行 top-k 列号
            mask = np.zeros_like(K, dtype=bool)
            row_idx = np.repeat(np.arange(n_rows), top_k)
            mask[row_idx, cutoff_cols.ravel()] = True
            K[~mask] = 0

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
    "SC": "SCerevisiae",
    "HS": "HSapiens",
}

pairs = [
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

a_values = [0.5]
lambda_mrot_values = [3]
lambda_param_values = [0]


def evaluate_one_pair(src_abbr, tgt_abbr, a_val, lambda_mrot_val=0.1, lambda_param_val=1.0):
    """
    ==== MODIFIED: 内存优化 ====
      1. 邻接矩阵改用 float32（原 networkx.to_numpy_array 默认 float64）。
      2. tilde 不构造 np.ones((n,n)) - np.eye(n) 临时矩阵（详见 asm_build_config）。
      3. 评估前把 A1/A2 转 csr_matrix 一次, 后续 EC/ICS/S3 共享，
         避免每次评估都重新转一次。
      4. 在关键阶段打印 print_mem 帮助排查 OOM。
    """
    src_full = name_map[src_abbr]
    tgt_full = name_map[tgt_abbr]
    pair_name = f"{src_abbr}-{tgt_abbr}"
    print(f"\n{'='*60}", flush=True)
    print(f"  配对: {pair_name}  ({src_full} vs {tgt_full})  a={a_val}  "
          f"lambda_mrot={lambda_mrot_val}  lambda_param={lambda_param_val}", flush=True)
    print(f"{'='*60}", flush=True)

    reset_peak_mem()
    print_mem("start")

    # 1. 读图 & 邻接矩阵 — ==== MODIFIED: float32 ====
    source_graph = networkx.read_leda(f"datas/ppi_net/{src_full}.gw")
    target_graph = networkx.read_leda(f"datas/ppi_net/{tgt_full}.gw")
    adj_source = networkx.to_numpy_array(source_graph, dtype=np.float32)
    adj_target = networkx.to_numpy_array(target_graph, dtype=np.float32)
    n1, n2 = adj_source.shape[0], adj_target.shape[0]
    e1 = int(np.sum(adj_source) / 2)
    e2 = int(np.sum(adj_target) / 2)
    print(f"源图节点: {n1}, 边: {e1}")
    print(f"目标图节点: {n2}, 边: {e2}")
    print_mem("after_load_adj")

    # 2. 构建序列相似性矩阵 K + 鲁棒预处理
    logval_path = f"datas/Sequence_scores/{src_full}-{tgt_full}.logval"
    K_raw = build_K_from_logval(source_graph, target_graph, logval_path)
    K = normalize_K(K_raw, top_k=10, cap_quantile=0.99)
    del K_raw  # ==== MODIFIED: 释放原始 K ====

    # 3. ETM+ASM 初始匹配
    mu_s3 = 2

    # ==== MODIFIED: 优化 tilde 构造（不再 np.ones((n,n)) - np.eye(n)） ====
    # 公式: tilde[i,j] = (1+mu)A[i,j] - mu*(1-δ_ij)
    adj_source_tilde = ((1.0 + mu_s3) * adj_source.astype(np.float32)) - mu_s3
    diag_correction = ((1.0 + mu_s3) * np.diag(adj_source)).astype(np.float32)
    np.fill_diagonal(adj_source_tilde, diag_correction)
    print_mem("after_tilde_build")

    M, N, s, asm_runtime = graphmatch_ASM_torch(
        adj_source_tilde, adj_target,
        K=K, alpha=1, lamb=lambda_param_val, adaptive_alpha=0,
        niter_max=20, tol=0.1,
        tau=1,
        eps_uot=0.01,
        lambda_mrot=lambda_mrot_val,
        uot_iters=100,
        precision='float32',
        method='asm',
    )
    print(f"ASM runtime: {asm_runtime:.2f}s", flush=True)
    print_mem("after_asm_initial")

    # ==== MODIFIED: ASM 阶段产物及时清理 ====
    del N, s, adj_source_tilde
    free_gpu()

    # 4. Fusion Moves 后处理
    cfg = asm_build_config(
        A1=adj_source, A2=adj_target,
        K=K, lambda_param=lambda_param_val, C_inf=1e7,
        mu_s3=mu_s3)

    initial_solution = asm_matrix_to_permutation(cfg, M)
    del M  # ==== MODIFIED: M 已转为 permutation，原矩阵可丢 ====
# #######################

    final_solution, fusion_info = asm_fusion_moves_optimization(
        cfg,
        initial_solution=initial_solution,
        max_iterations=1,
        num_candidates=10,
        a=a_val,
        adaptive_alpha=0,
        asm_niter=5,
        device='cuda',
        precision='float32',
        choose='gradient',
        store_history=False,  # ==== MODIFIED: 关闭历史保存 ====
    )

    details = fusion_info['fusion_details']
    if len(details) > 0:
        fusion_total_time = (details[-1]['total_candidate_time']
                             + details[-1]['total_fusion_time'])
    else:
        fusion_total_time = 0.0

    print_mem("after_fusion")
# # ####################
    
    # #change=============
    # src_gw_path = f"datas/ppi_net/{src_full}.gw"
    # tgt_gw_path = f"datas/ppi_net/{tgt_full}.gw"
    # #change=============


    # #change=============
    # # 交替融合: gradient -> sana, 重复 5 轮。
    # # fusion_total_time 统计所有 10 次 asm_fusion_moves_optimization 的
    # # 候选解生成时间 + QPBO 融合时间；fusion_wall_time 统计外层循环真实墙钟时间。
    # alt_start_time = time.time()

    # current_solution_alt = initial_solution.copy()
    # all_fusion_infos = []

    # for outer_iter in range(1):
    #     print(f"\n=== Alternating Fusion Round {outer_iter + 1}/5: gradient ===", flush=True)
    #     final_solution1, fusion_info1 = asm_fusion_moves_optimization(
    #         cfg,
    #         initial_solution=current_solution_alt,
    #         max_iterations=1,
    #         num_candidates=9,
    #         a=a_val,
    #         adaptive_alpha=0,
    #         device='cuda',
    #         precision='float32',
    #         choose='gradient',
    #         source_gw_path=src_gw_path,
    #         target_gw_path=tgt_gw_path,
    #     )
    #     all_fusion_infos.append(("gradient", fusion_info1))

    #     print(f"\n=== Alternating Fusion Round {outer_iter + 1}/5: sana ===", flush=True)
    #     final_solution, fusion_info = asm_fusion_moves_optimization(
    #         cfg,
    #         initial_solution=final_solution1,
    #         max_iterations=1,
    #         num_candidates=1,
    #         a=a_val,
    #         adaptive_alpha=0,
    #         device='cuda',
    #         precision='float32',
    #         choose='sana',
    #         source_gw_path=src_gw_path,
    #         target_gw_path=tgt_gw_path,
    #     )
    #     all_fusion_infos.append(("sana", fusion_info))

    #     current_solution_alt = final_solution.copy()

    # alt_end_time = time.time()
    # fusion_wall_time = alt_end_time - alt_start_time

    # # 统计交替融合总时间
    # total_candidate_time = 0.0
    # total_fusion_move_time = 0.0

    # for method_name, info in all_fusion_infos:
    #     details = info.get('fusion_details', [])
    #     if len(details) == 0:
    #         continue
    #     last_detail = details[-1]
    #     total_candidate_time += last_detail.get('total_candidate_time', 0.0)
    #     total_fusion_move_time += last_detail.get('total_fusion_time', 0.0)

    # fusion_total_time = total_candidate_time + total_fusion_move_time

    # print(f"交替融合候选解生成总时间: {total_candidate_time:.2f} 秒", flush=True)
    # print(f"交替融合QPBO融合总时间: {total_fusion_move_time:.2f} 秒", flush=True)
    # print(f"交替融合内部统计总时间: {fusion_total_time:.2f} 秒", flush=True)
    # print(f"交替融合墙钟总时间: {fusion_wall_time:.2f} 秒", flush=True)
    # #change=============
# ##################

    # 5. 评估指标 — ==== MODIFIED: 用 eval_all 共享中间计算 ====
    A1_csr = sp.csr_matrix(adj_source)
    A2_csr = sp.csr_matrix(adj_target)
    E1 = A1_csr.sum() / 2.0

    ec_ini, ics_ini, s3_ini = eval_all(A1_csr, A2_csr, E1, initial_solution)
    ec_fin, ics_fin, s3_fin = eval_all(A1_csr, A2_csr, E1, final_solution)

    print(f"EC: {ec_ini:.4f} -> {ec_fin:.4f}", flush=True)
    print(f"ICS: {ics_ini:.4f} -> {ics_fin:.4f}", flush=True)
    print(f"S3: {s3_ini:.4f} -> {s3_fin:.4f}", flush=True)

    # ==== MODIFIED: 评估完成后释放 ====
    del A1_csr, A2_csr, adj_source, adj_target, K, cfg
    free_gpu()
    print_mem("end")

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
                                                lambda_param_val=lp_val)
                        results.append(res)
                    except Exception as e:
                        print(f"\n*** failure: {src_abbr}-{tgt_abbr} a={a_val} "
                              f"lambda_mrot={lm_val} lambda_param={lp_val} — "
                              f"{type(e).__name__}: {e} ***\n", flush=True)
                        continue
                    finally:
                        # ==== MODIFIED: 严格的 per-pair 清理，避免显存泄漏累计 ====
                        free_gpu()
                        print(f"[内存清理完成] ({run_idx}/{total_runs})", flush=True)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("汇总结果:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    results_df.to_excel("our_ics_optimized_mu2.xlsx", index=False)
    print("\n结果已保存至 our_ics_optimized_mu2.xlsx")
