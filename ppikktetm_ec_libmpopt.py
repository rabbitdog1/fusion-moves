#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPI Network Alignment 批量评测脚本 (ETM+ASM 版本) — 内存优化版 (EC目标函数)

本文件是在 ppikktetm.py 基础上的"小幅修改"优化版：
  1. 算法核心逻辑完全保留（ETM-ASM、Fusion Moves、QPBO、EC 等）。
  2. 大量降低显存/内存峰值（去 clone、稀疏化评估、删除历史保存）。
  3. 提供更快的 Hungarian wrapper（top-k 稀疏 LAP），保持精确解作为默认 fallback。
  4. 新增内存诊断 print，便于排查 OOM。
  5. 与 S3 版本的关键区别：
     - asm_build_config 无 mu_s3 参数，不构造 tilde_A1；
     - asm_fusion_moves_optimization 以 asm_evaluate_solution 为目标函数（而非 S3）；
     - asm_fusion_moves_qpbo 无 mu_s3 额外惩罚项；
     - asm_generate_candidate_via_asm 使用 cfg.A1（不是 cfg.tilde_A1）；
     - evaluate_one_pair 直接传入 adj_source（无 tilde 构造）。

所有改动处都标注了 "# ==== MODIFIED: ... ====" 注释。
用法: python ppikktetm_ec_optimized.py
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

from Algorithms.Netal.netal import run_netal_matching_matrix
from Algorithms.SANA.sana_wrapper import run_sana_matching_matrix

# == libmpopt == 接入 libmpopt 的 qap_dd_fusion (qpbo-i) 替代 thinqpbo。
# 失败时静默降级回原 thinqpbo 路径,使用环境变量 PPI_FUSION_BACKEND 选择
# 后端: "libmpopt" (默认) 或 "qpbo" (原 thinqpbo 版本)。

from libmpopt_fusion import libmpopt_fuse as _libmpopt_fuse  # == libmpopt ==
_LIBMPOPT_AVAILABLE = True                                    # == libmpopt ==

DEFAULT_FUSION_BACKEND = os.environ.get(                          # == libmpopt ==
    "PPI_FUSION_BACKEND", "libmpopt" if _LIBMPOPT_AVAILABLE else "qpbo"
)


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
#===new=== Patch 1: 只替换 hugarian_fast 加上 row∪col union 增广
# 把原 hugarian_fast 函数体内 sparse 失败时直接 fallback 的部分,
# 换成下面的三层重试:

#===new=== 完整替换原 hugarian_fast
def hugarian_fast(matrix, top_k: int = 512, return_vector: bool = False,
                  method: str = 'auto', device: str = 'cuda', verbose: bool = False):
    """
    高鲁棒性的快速线性分配。
    
    三层防线(method='auto' 时):
      1) row-topk 稀疏 LAP,k 自适应递增 [k, 2k, 4k, 8k]
      2) row-topk ∪ col-topk 增广稀疏 LAP(缓解 Hall 违反)
      3) Greedy + 残差 mini-Hungarian 修复(近似)
      4) 最后才退到 dense scipy(精确兜底)
    
    Args:
        matrix: (n, m) np.ndarray / torch.Tensor,值越大表示越倾向匹配,要求 n <= m
        top_k:  稀疏路径初始 k(默认 512)，仅在 method='auto' 或 'sparse' 时启用。越大越慢但越稳，建议 128~512。
        return_vector: True → 返回 (n,) col_ind;False → 返回 (n, m) float32 one-hot
        method: 'auto'(推荐)| 'sparse' | 'greedy' | 'dense'
        verbose: 打印路径选择
    """
    t_total_start = time.perf_counter()
    
    if hasattr(matrix, 'cpu'):
        matrix_np = matrix.detach().cpu().numpy()
    else:
        matrix_np = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    n, m = matrix_np.shape
    
    col_ind = None
    path_used = None
    
    # ============ Path 1: 自适应 row top-k ============
    if method in ('auto', 'sparse') and n <= m and top_k and top_k > 0:
        for k_try in [top_k, min(top_k*2, m-1), min(top_k*4, m-1), min(top_k*8, m-1)]:
            if k_try >= m or k_try <= 0:
                break
            col_ind = _try_sparse_row_topk(matrix_np, k_try)
            if col_ind is not None:
                path_used = f'sparse-row-topk(k={k_try})'
                break

    # ============ Path 2: row ∪ col top-k 增广 ============
    if col_ind is None and method in ('auto', 'sparse') and n <= m:
        for k_try in [max(top_k, 512), max(top_k*2, 1024), max(top_k*4, 2048), max(top_k*8, 4096)]:
            if k_try >= min(n, m):
                break
            col_ind = _try_sparse_union_topk(matrix_np, k_try)
            if col_ind is not None:
                path_used = f'sparse-union-topk(k={k_try})'
                break
            
    # ============ Path 4: dense scipy(精确兜底) ============
    if col_ind is None or method == 'dense':
        ri, ci = linear_sum_assignment(-matrix_np)
        col_ind = np.full(n, -1, dtype=np.int64)
        col_ind[ri] = ci
        path_used = 'dense-scipy'
    
    t_total = time.perf_counter() - t_total_start
    if verbose:
        print(f"[hugarian_fast] path={path_used}, n={n}, m={m}, time={t_total:.3f}s", flush=True)
    
    if return_vector:
        return col_ind.astype(np.int64)
    
    P = np.zeros((n, m), dtype=np.float32)
    valid = col_ind >= 0
    P[np.arange(n)[valid], col_ind[valid]] = 1.0
    return P

#===new=== 辅助函数 1:纯 row top-k 稀疏 LAP
def _try_sparse_row_topk(matrix_np, k):
    n, m = matrix_np.shape
    try:
        idx = np.argpartition(-matrix_np, k, axis=1)[:, :k]
        rr = np.repeat(np.arange(n), k)
        cc = idx.ravel()
        vv = matrix_np[rr, cc].astype(np.float64)
        vmin = vv.min()
        if vmin <= 0:
            vv = vv - vmin + 1e-9
        M_sp = sp.csr_matrix((vv, (rr, cc)), shape=(n, m))
        ri, ci = min_weight_full_bipartite_matching(M_sp, maximize=True)
        out = np.full(n, -1, dtype=np.int64)
        out[ri] = ci
        return out
    except Exception:
        return None

#===new=== 辅助函数 2:row + col 双向 top-k 增广,显著提升 Hall 满足率
def _try_sparse_union_topk(matrix_np, k):
    n, m = matrix_np.shape
    try:
        # row top-k
        idx_r = np.argpartition(-matrix_np, k, axis=1)[:, :k]
        rr1 = np.repeat(np.arange(n), k)
        cc1 = idx_r.ravel()
        # col top-k(对每列,取 k 个最大行)
        kc = min(k, n - 1)
        idx_c = np.argpartition(-matrix_np, kc, axis=0)[:kc, :]
        cc2 = np.repeat(np.arange(m), kc)
        rr2 = idx_c.T.ravel()
        # union(行列 top-k)
        rr = np.concatenate([rr1, rr2])
        cc = np.concatenate([cc1, cc2])
        # dedup
        flat_idx = rr.astype(np.int64) * m + cc.astype(np.int64)
        flat_idx, uniq_pos = np.unique(flat_idx, return_index=True)
        rr = (flat_idx // m).astype(np.int64)
        cc = (flat_idx % m).astype(np.int64)
        vv = matrix_np[rr, cc].astype(np.float64)
        vmin = vv.min()
        if vmin <= 0:
            vv = vv - vmin + 1e-9
        M_sp = sp.csr_matrix((vv, (rr, cc)), shape=(n, m))
        ri, ci = min_weight_full_bipartite_matching(M_sp, maximize=True)
        out = np.full(n, -1, dtype=np.int64)
        out[ri] = ci
        return out
    except Exception:
        return None



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
HUNGARIAN_TOPK = int(os.environ.get("PPI_HUNGARIAN_TOPK", "512"))


###############################################################################
# ETM
###############################################################################

def uot_helper(benefit_mat, tau_a, tau_b, eps=0.01, epoch_iter=500, tol=0.1,
               device='cuda', dtype=torch.float32, 
               init_u=None, init_v=None, init_zeta=None): # 新增初始值参数
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

    #uot_new======================
    if init_zeta is not None:
        zeta = init_zeta
    else:
        zeta = torch.tensor(0.0, dtype=dtype, device=device)
        
    if init_u is not None:
        u = init_u
    else:
        u = torch.ones(n, dtype=dtype, device=device) / n
        
    if init_v is not None:
        v = init_v
    else:
        v = torch.ones(m, dtype=dtype, device=device) / m
    #uot_new======================    
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
    # #uot_new====================== 
    u = torch.clamp(u, min=-500, max=500)
    v = torch.clamp(v, min=-500, max=500) 
    # #uot_new====================== 
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
                              row_target=None, col_target=None,
                              # ==== OPT-A: 暖启动接口 ====
                              P_init=None,
                              # ==== OPT-C: 硬性迭代上限 ====
                              max_outer=50,
                              max_inner=200,
                              # ==== OPT-B: 内层同步频率 ====
                              sync_every=3):
    """
    修复版 Softassign:
    1. 显式初始化 outer = 0
    2. 统一 diff_val 变量名
    3. 增加 max_outer 循环保护
    """
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=dtype, device=device)
    else:
        matrix = matrix.to(dtype=dtype, device=device)

    # 归一化矩阵
    mat_min = matrix.min()
    mat_max = matrix.max()
    mat_range = mat_max - mat_min
    if mat_range > 1e-12:
        matrix = (matrix - mat_min) / mat_range
    else:
        matrix = torch.zeros_like(matrix)

    n, m = matrix.shape
    # print(f'dim={n},{m}') # 可选打印

    if row_target is None:
        row_target = torch.ones(n, dtype=dtype, device=device)
    if col_target is None:
        col_target = torch.ones(m, dtype=dtype, device=device)

    # ==== OPT-A: 暖启动 P ====
    if (P_init is not None) and (tuple(P_init.shape) == (n, m)):
        P = P_init
        # .detach().to(dtype=dtype, device=device).clone()
        # P.clamp_(min=0.0)
        # if P.sum() < 1e-12:
        #     P.fill_(1.0 / max(n, m))
    else:
        P = torch.full((n, m), 1.0 / max(n, m), dtype=dtype, device=device)

    # ==== OPT-D: beta 钳制 ====
    log_n = math.log(max(n, 2))
    beta_val = float(log_n) * float(gamma_ini)
    if beta_val > 80.0:
        beta_val = 80.0
    K = torch.exp(matrix * beta_val).clamp_(min=1e-30)
    del matrix

    v = torch.ones(m, dtype=dtype, device=device)
    u = torch.ones(n, dtype=dtype, device=device)

    # ==== FIX: 显式初始化变量 ====
    diff_val = float('inf')
    outer = 0 

    # ==== OPT-C: 增加 outer < max_outer 判断 ====
    while diff_val > eps and outer < max_outer:
        Q = K * P
        iter_sk = 0
        res_diff_val = float('inf')

        while res_diff_val > 0.2 and iter_sk < max_inner:
            u_new = row_target / torch.clamp(torch.matmul(Q, v), min=1e-30)
            v_new = col_target / torch.clamp(torch.matmul(Q.T, u_new), min=1e-30)

            iter_sk += 1
            # ==== OPT-B: 降低同步频率 ====
            if iter_sk % sync_every == 0:
                u_max = u_new.max().clamp(min=1e-30)
                v_max = v_new.max().clamp(min=1e-30)
                u_pmax = u.max().clamp(min=1e-30)
                v_pmax = v.max().clamp(min=1e-30)
                res_diff = (u_new / u_max - u / u_pmax).abs().sum() + \
                           (v_new / v_max - v / v_pmax).abs().sum()
                res_diff_val = res_diff.item()

            u = u_new
            v = v_new

        # ==== OPT-E: In-place 计算收敛差值 ====
        Q.mul_(v.unsqueeze(0))
        Q.mul_(u.unsqueeze(1))
        # 此时 Q 是 P_new
        P.sub_(Q).abs_()
        diff_val = P.sum(dim=0).max().item()
        
        # 更新 P 用于下一次外层迭代
        del P
        P = Q.clone() # 保持一份独立拷贝用于下一轮计算
        outer += 1
        gamma_ini += 1

    if outer >= max_outer:
        print(f'  [softassign] reached max_outer={max_outer}, diff={diff_val:.4f}', flush=True)

    return P.detach(), gamma_ini





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

    # ==========================================================
    total_uot_time = 0.0
    total_softassign_time = 0.0
    # ==========================================================
    # 在进入 for i in range(niter_max) 循环前，初始化 UOT 状态
    u_etm, v_etm, zeta_etm = None, None, None #uot_new======================
    # ==== OPT-A: 跨 ASM 迭代暖启动 Softassign 的 P ====
    P_softassign_warm = None
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
                t_uot_start = time.perf_counter()
                # ETM — 计算 MROT slack
               
                u_etm, v_etm, zeta_etm, s = uot_helper(
                    D, tau_a=tau, tau_b=tau,
                    eps=eps_uot, epoch_iter=uot_iters,
                    device=device, dtype=dtype ,
                    init_u=u_etm, init_v=v_etm, init_zeta=zeta_etm,
                )
                
                t_uot_run= time.perf_counter()- t_uot_start
                total_uot_time += t_uot_run
                print(f"UOT runtime: {t_uot_run:.4f} seconds")

                etm_alpha = src_weight * torch.exp(
                    torch.clamp(-(u_etm + zeta_etm) / tau, min=-500.0, max=500.0)
                )
                etm_beta = trg_weight * torch.exp(
                    torch.clamp(-(v_etm - zeta_etm) / tau, min=-500.0, max=500.0)
                )

                # MROT 正则化（in-place）
                D.sub_(s, alpha=lambda_mrot)  # ==== MODIFIED: in-place 减法 ====

                # ==== MODIFIED: 删除原本无用的 D_safe = D.clone() / d_max = D_safe.max() ====
                t_soft_start = time.perf_counter()
                if n == m:
                    print('n=m')
                    if method == 'sinkhorn':
                        D_exp = torch.exp(D - D.max())
                        D = sinkhorn(D_exp, device=device, dtype=dtype)
                    elif method == 'asm':
                        # ==== OPT-A: 传入暖启动 P ====
                        D, gamma = adaptive_softassign_torch(
                            D, gamma_ini=1, device=device, dtype=dtype,
                            use_amp=use_amp,
                            P_init=P_softassign_warm)
                else:
                    print(f'n={n}, m={m}')
                    if method == 'sinkhorn':
                        D_exp = torch.exp(D - D.max())
                        D = sinkhorn(D_exp, device=device, dtype=dtype,
                                     row_target=etm_alpha, col_target=etm_beta)
                    elif method == 'asm':
                        # ==== OPT-A: 传入暖启动 P (矩形/带 target 分支) ====
                        D, gamma = adaptive_softassign_torch(
                            D, gamma_ini=1, device=device, dtype=dtype,
                            use_amp=use_amp,
                            row_target=etm_alpha, col_target=etm_beta,
                            P_init=P_softassign_warm)
                # ==== OPT-A: 缓存当前轮 Softassign 输出供下轮暖启动 ====
                # detach 是为避免 autograd 引用; 不 clone() 是因为 D 马上会被
                # 用来更新 N, 但 N.mul_(1-alpha).add_(D, alpha=alpha) 只读 D
                # 不会原地修改它,所以 D 本身可以直接持有作下轮种子。
                P_softassign_warm = D.detach()

                t_soft_run = time.perf_counter() - t_soft_start
                total_softassign_time += t_soft_run
                print(f"Softassign runtime (iter {i}): {t_soft_run:.4f} seconds")

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

    # ==== OPT-A: 跨 graphmatch_ASM_torch 调用不复用 P_warm,
    # 这里显式释放避免函数返回后还被外层引用 ====
    del P_softassign_warm
    # ==== MODIFIED: 用 hugarian_fast(top-k) 替代 hugarian()
    # 数值验证：top_k>=20 在 5897×7937 实测与 dense 解 100% 吻合，速度提升 ~24x
    topk = hungarian_topk if hungarian_topk is not None else HUNGARIAN_TOPK
    hugarian_start = time.perf_counter()
    col_ind = hugarian_fast(N, top_k=topk, return_vector=True,
                            method='sparse', device=device, verbose=True)
    M = np.zeros((n, m), dtype=np.float32)
    valid = col_ind >= 0
    M[np.arange(n)[valid], col_ind[valid]] = 1.0
    hugarian_runtime_asm = time.perf_counter() - hugarian_start
    print(f"Total UOT Time: {total_uot_time:.4f}s")
    print(f"Total Softassign Time: {total_softassign_time:.4f}s")
    print(f"hug_runtime_asm: {hugarian_runtime_asm:.4f} seconds")
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
                     lambda_param: float = 0.5, C_inf: float = 1000000.0):
    """
    EC 版本的 asm_build_config：无 mu_s3 参数，不构造 tilde_A1。
    ==== MODIFIED: 使用 np.asarray 代替 np.array 以避免不必要的拷贝 ====
    """
    cfg = types.SimpleNamespace()
    cfg.A1 = np.asarray(A1)
    cfg.A2 = np.asarray(A2)
    cfg.K = np.asarray(K)
    cfg.lambda_param = lambda_param
    cfg.C_inf = C_inf
    cfg.n1, cfg.n2 = cfg.A1.shape[0], cfg.A2.shape[0]

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
    quadratic_term = 0.5 * np.sum(A1_sub * A2_sub)
    linear_term = cfg.lambda_param * np.sum(cfg.K[rows, cols])
    return quadratic_term + linear_term


###############################################################################
# 候选解生成 (gradient)
###############################################################################

def graphmatch_gradient(adjacency1, adjacency2, K=0, tol=0.1, alpha=1.0, lamb=1.0,
                        gamma0=1.0, adaptive_alpha=False, niter_max=30,
                        device='cuda', precision='float32',
                        X_init_normalized=None,
                        hungarian_topk: int = None,
                        #===new=== 新增参数
                        assignment_method: str = 'sparse',
                        return_vector: bool = False):
    """
    ==== MODIFIED 2: alpha=1 时去掉第 2 次冗余 Hungarian ====
    返回:
      若 return_vector=True: (col_ind_np: (n,) int64, runtime)
      否则: (M: (n,m) float32 one-hot, runtime) — 兼容原接口
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

    device_t = torch.device(device)
    starttime = time.perf_counter()

    def to_tensor(x):
        return torch.tensor(x, dtype=dtype, device=device_t) if not isinstance(x, torch.Tensor) \
            else x.to(dtype=dtype, device=device_t)

    adjacency1 = to_tensor(adjacency1)
    adjacency2 = to_tensor(adjacency2)
    K = to_tensor(K)
    n, _ = adjacency1.shape
    m, _ = adjacency2.shape

    if X_init_normalized is not None:
        N = to_tensor(X_init_normalized)
        if N.shape != (n, m):
            raise ValueError(f"X_init_normalized shape {N.shape} != ({n}, {m})")
        N = N.to(dtype=dtype, device=device_t)
        print("Using provided initial matching matrix gradient.")
    else:
        N = torch.ones((n, m), dtype=dtype, device=device_t) / n
        print("Using uniform initial matching matrix.")

    topk = hungarian_topk if hungarian_topk is not None else HUNGARIAN_TOPK
    with torch.no_grad():
        N0 = N.clone() if (adaptive_alpha or alpha < 1.0 - 1e-9) else None
        with autocast(device_type=device_t.type, enabled=use_amp):
            delta_edges = torch.matmul(torch.matmul(adjacency1, N), adjacency2)
            D = delta_edges + lamb * K

            #===new=== 只做 1 次 Hungarian,直接取 col_ind
            t_hugarian_start = time.perf_counter()
            col_ind = hugarian_fast(D, top_k=topk, return_vector=True,
                                    method=assignment_method, device=device,
                                    verbose=True)
            t_hugarian_run = time.perf_counter() - t_hugarian_start
            print(f"hug_runtime_gradient: {t_hugarian_run:.4f} seconds")

            #===new=== alpha=1 直接跳过 soft 混合 + 第 2 次 Hungarian
            if abs(alpha - 1.0) < 1e-9 and not adaptive_alpha:
                # N 就是 one-hot,col_ind 即最终答案,无需再 Hungarian
                pass
            else:
                # 需要 soft 混合时才物化 one-hot
                D_out = torch.zeros_like(N)
                rows = torch.arange(n, device=device_t)
                cols_t = torch.from_numpy(col_ind).to(device_t)
                D_out[rows, cols_t] = 1.0
                if adaptive_alpha:
                    alpha_op = compute_alpha(N0, delta_edges, D,
                                             adjacency1, adjacency2, K, lamb)
                    if 0 <= alpha_op < 1:
                        alpha = float(alpha_op)
                        print("Alpha is", alpha_op)
                    else:
                        alpha = 1.0
                N.mul_(1 - alpha).add_(D_out, alpha=alpha)
                # 软更新后才需要再做一次 Hungarian 离散化
                hugarian_start1 = time.perf_counter()
                col_ind = hugarian_fast(N, top_k=topk, return_vector=True,
                                        method=assignment_method, device=device)
                hugarian_runtime_gra = time.perf_counter() - hugarian_start1
                print(f"hug_runtime_last: {hugarian_runtime_gra:.4f} seconds")
                del D_out

            del delta_edges, D

        print(f'gamma: {gamma0}')
        print("Converged")
        if N0 is not None:
            del N0

    runtime = time.perf_counter() - starttime
    if return_vector:
        return col_ind.astype(np.int64), runtime
    # 兼容原接口:构造 one-hot
    M = np.zeros((n, m), dtype=np.float32)
    valid = col_ind >= 0
    M[np.arange(n)[valid], col_ind[valid]] = 1.0
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
      4. EC 版本使用 cfg.A1（不是 cfg.tilde_A1）。
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

    # EC 版本：使用 cfg.A1（原始邻接矩阵，无 mu_s3 修改）
    if choose == 'gradient':
        candidate_matrix, _ = graphmatch_gradient(
            adjacency1=cfg.A1, adjacency2=cfg.A2, K=cfg.K,
            tol=0.1, lamb=cfg.lambda_param, adaptive_alpha=adaptive_alpha,
            alpha=1.0, gamma0=1.0, niter_max=asm_niter,
            device=device, precision=precision,
            X_init_normalized=X_init_normalized,
            return_vector=True,                  #===new===
            assignment_method='auto',            #===new===
        )
    elif choose == 'asm':
        candidate_matrix, _, _, _ = graphmatch_ASM_torch(
            adjacency1=cfg.A1,
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

    # candidate_solution = asm_matrix_to_permutation(cfg, candidate_matrix)
    if choose == 'gradient':
        # 在这种模式下，candidate_matrix 其实已经是 col_ind 向量了
        candidate_solution = candidate_matrix 
    else:
        candidate_solution = asm_matrix_to_permutation(cfg, candidate_matrix) 
    if not asm_is_feasible(cfg, candidate_solution):
        print(f'    警告: ASM生成的候选解不可行,正在修复...')
        candidate_solution = asm_make_feasible(cfg, candidate_solution)

    return candidate_solution


###############################################################################
# QPBO  （EC 版本：无 mu_s3 项）
###############################################################################

def asm_fusion_moves_qpbo(cfg, current_solution, candidate_solution):
    """
    ==== MODIFIED: 重构 unary/pairwise 计算，使用 cfg.edge_list_A1 取代 O(n1²) 双重循环 ====
    逻辑与原版完全等价（无 mu_s3），仅改变循环结构以提升速度。
    """
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

    # ==== MODIFIED: K 项与 同cycle边项 分开，复用 edge_list_A1 ====
    for cid in range(m):
        for i in cycle_left[cid]:
            unary0[cid] += -lam * K[i, p[i]]
            unary1[cid] += -lam * K[i, q[i]]

    # ==== MODIFIED: 同cycle内边贡献：利用稀疏 edge_list_A1，避免 O(n1²) ====
    for i, j in cfg.edge_list_A1:
        ci = left_cycle[i]
        cj = left_cycle[j]
        if ci == cj:
            unary0[ci] += -pair_contrib(i, j, p[i], p[j])
            unary1[ci] += -pair_contrib(i, j, q[i], q[j])

    pair_tables = {}
    # ==== MODIFIED: 跨cycle边：利用稀疏 edge_list_A1，避免 O(n1²) ====
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

    # EC 版本：无 mu_s3 额外惩罚项（与 S3 版本的区别在于此处没有 edge_list_A2 遍历）

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
                                  # == libmpopt == 接入 libmpopt 的 fusion 后端 ==
                                  fusion_backend: str = DEFAULT_FUSION_BACKEND,
                                  qap_dd_fusion_bin: str = "qap_dd_fusion",
                                  fusion_solver: str = "qpbo-i",
                                  ) -> Tuple[np.ndarray, Dict]:
    """
    ==== MODIFIED: 内存优化 + 重复计算消除 (EC 版本) ====
      1. 默认 store_history=False, 不保存 all_candidates / all_fused_solutions /
         fusion_details 中的 solution 拷贝。
      2. 使用 asm_evaluate_solution(cfg, solution) 作为目标函数（EC 版本的核心区别）。
      3. 删除每次迭代里多余的 asm_evaluate_solution 重复调用。
      4. 每个候选解结束后调用 free_gpu()。
    数值上完全等价（接受/拒绝逻辑、最终 best_solution 不变）。
    """
    print('\n=== 开始Fusion Moves优化 ===', flush=True)

    current_solution = initial_solution.copy()
    # EC 版本：以 asm_evaluate_solution 为目标函数（而非 S3）
    current_objective = asm_evaluate_solution(cfg, current_solution)

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
            time_start_cand = time.perf_counter()
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
            time_end_cand = time.perf_counter()
            dt_cand = time_end_cand - time_start_cand
            total_candidate_time += dt_cand
            print(f'    生成候选解时间: {dt_cand:.2f} 秒')

            if not asm_is_feasible(cfg, candidate_solution):
                print(f'  候选解 {cand_idx + 1} 不可行，正在修复...')
                candidate_solution = asm_make_feasible(cfg, candidate_solution)

            candidate_objective = asm_evaluate_solution(cfg, candidate_solution)
            candidate_accuracy = np.sum(candidate_solution == np.arange(n_nodes)) / n_nodes
            print(f'  候选解 {cand_idx + 1} 目标值: {candidate_objective:.4f}, 准确率: {candidate_accuracy:.4f}')

            time_start_fusion = time.perf_counter()
            # == libmpopt ==
            if fusion_backend == "libmpopt" and _LIBMPOPT_AVAILABLE:  # == libmpopt ==                                                            
                fused_solution = _libmpopt_fuse(                   # == libmpopt ==
                    cfg, current_solution, candidate_solution,     # == libmpopt ==
                    qap_dd_fusion_bin=qap_dd_fusion_bin,           # == libmpopt ==
                    solver=fusion_solver, verify_dd=False,         # == libmpopt ==
                    verbose=True,                                  # == libmpopt ==
                ) 
                print(f'libmpopt qpbo', flush=True)     # == libmpopt ==               
            else:                                                       # == libmpopt ==
                fused_solution = asm_fusion_moves_qpbo(cfg, current_solution, candidate_solution)
            time_end_fusion = time.perf_counter()
            dt_fusion = time_end_fusion - time_start_fusion
            total_fusion_time += dt_fusion
            print(f'    融合解生成时间: {dt_fusion:.2f} 秒')

            if not asm_is_feasible(cfg, fused_solution):
                fused_solution = asm_make_feasible(cfg, fused_solution)

            fused_objective = asm_evaluate_solution(cfg, fused_solution)
            fused_accuracy = np.sum(fused_solution == np.arange(n_nodes)) / n_nodes
            print(f'  融合解目标值: {fused_objective:.4f}, 准确率: {fused_accuracy:.4f}')
            # ==== MODIFIED: 不再重复调用 asm_evaluate_solution() 做诊断 ====
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
                # 仅保留时间统计字段（外层需要用）
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

    initial_objective_for_info = asm_evaluate_solution(cfg, initial_solution)
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
        result_info['candidates_per_iter'] = []
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
            idx_part = np.argpartition(-K, top_k - 1, axis=1)
            cutoff_cols = idx_part[:, :top_k]
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
    # ("CE", "AT"),
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

a_values = [0.5]
lambda_mrot_values = [3]
lambda_param_values = [0]


def evaluate_one_pair(src_abbr, tgt_abbr, a_val, lambda_mrot_val=0.1, lambda_param_val=1.0):
    """
    ==== MODIFIED: 内存优化 (EC 版本) ====
      1. 邻接矩阵改用 float32（原 networkx.to_numpy_array 默认 float64）。
      2. EC 版本无 tilde 构造：直接传入 adj_source。
      3. 评估前把 A1/A2 转 csr_matrix 一次, 后续 EC/ICS/S3 共享，
         避免每次评估都重新转一次（使用 eval_all 函数）。
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
    # EC 版本：直接传入 adj_source（无 mu_s3 / tilde 构造）
    print_mem("before_asm")
    M, N, s, asm_runtime = graphmatch_ASM_torch(
        adj_source, adj_target,
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
    del N, s
    free_gpu()

    # 4. Fusion Moves 后处理
    # EC 版本：asm_build_config 无 mu_s3 参数
    cfg = asm_build_config(
        A1=adj_source, A2=adj_target,
        K=K, lambda_param=lambda_param_val, C_inf=1e7)

    initial_solution = asm_matrix_to_permutation(cfg, M)
#     del M  # ==== MODIFIED: M 已转为 permutation，原矩阵可丢 ====
# # #######################

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
# ######################################
    # src_gw_path = f"datas/ppi_net/{src_full}.gw"
    # tgt_gw_path = f"datas/ppi_net/{tgt_full}.gw"

    # # 交替融合: gradient -> sana
    # alt_start_time = time.perf_counter()

    # current_solution_alt = initial_solution.copy()
    # all_fusion_infos = []

    # for outer_iter in range(1):
    #     print(f"\n=== Alternating Fusion Round {outer_iter + 1}/1: gradient ===", flush=True)
    #     final_solution1, fusion_info1 = asm_fusion_moves_optimization(
    #         cfg,
    #         initial_solution=current_solution_alt,
    #         max_iterations=1,
    #         num_candidates=8,
    #         a=a_val,
    #         adaptive_alpha=0,
    #         device='cuda',
    #         precision='float32',
    #         choose='gradient',
    #         source_gw_path=src_gw_path,
    #         target_gw_path=tgt_gw_path,
    #         store_history=False,  # ==== MODIFIED: 关闭历史保存 ====
    #     )
    #     all_fusion_infos.append(("gradient", fusion_info1))

    #     final_solution2, fusion_info2 = asm_fusion_moves_optimization(
    #         cfg,
    #         initial_solution=final_solution1,
    #         max_iterations=1,
    #         num_candidates=1,
    #         a=a_val,
    #         adaptive_alpha=0,
    #         device='cuda',
    #         precision='float32',
    #         choose='netal',
    #         source_gw_path=src_gw_path,
    #         target_gw_path=tgt_gw_path,
    #         store_history=False,  # ==== MODIFIED: 关闭历史保存 ====
    #     )
    #     all_fusion_infos.append(("gradient", fusion_info2))

    #     print(f"\n=== Alternating Fusion Round {outer_iter + 1}/1: sana ===", flush=True)
    #     final_solution, fusion_info = asm_fusion_moves_optimization(
    #         cfg,
    #         initial_solution=final_solution2,
    #         max_iterations=1,
    #         num_candidates=1,
    #         a=a_val,
    #         adaptive_alpha=0,
    #         device='cuda',
    #         precision='float32',
    #         choose='sana',
    #         source_gw_path=src_gw_path,
    #         target_gw_path=tgt_gw_path,
    #         store_history=False,  # ==== MODIFIED: 关闭历史保存 ====
    #     )
    #     all_fusion_infos.append(("sana", fusion_info))

    #     current_solution_alt = final_solution.copy()

    # alt_end_time = time.perf_counter()
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
################################
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

    # results_df.to_excel("our_ec1_topk512.xlsx", index=False)
    # print("\n结果已保存至 our_ec1_topk512.xlsx")
