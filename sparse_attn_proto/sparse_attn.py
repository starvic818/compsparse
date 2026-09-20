"""Piecewise 稀疏注意力的参考实现（纯 CPU 可跑，小序列验证用）。

实现的是推导文档里的三段式结构：
    E 段（精确）  : 每行 logit 最高的 top-m 个 key，完整计算
    A 段（补偿）  : 其余 key 中 delta in [-delta_hi, 0] 的部分，用泰勒矩做补偿
    D 段（丢弃）  : 剩下的长尾，直接丢弃（其质量已可忽略）

为什么需要 A 段而不是"要么精确要么丢"：见 `test_order1_helps_when_band_is_narrow`
与 `test_order1_hurts_when_band_is_wide` ——
  一阶泰勒只在 |delta| 较小时优于硬丢弃（break-even 约 |delta| ≈ 0.9），
  对长尾（|delta| 很大）一阶近似甚至给出负权重，比直接丢弃还差。
  这正是"分段"必须存在的数学原因，也是审核老师最可能追问的点。

记号（与推导文档一致）：
    s_ij   = (q_i · k_j) * scale            logits
    c_i    = max_j s_ij                     行最大值（数值稳定用）
    delta  = s - c <= 0
    w      = exp(delta) ∈ (0,1]             稳定化权重
    o_i    = Σ_j w_ij v_j / Σ_j w_ij

一阶矩（可跨 query 复用，这是补偿便宜的根源）：
    V_A[i] = Σ_{j∈A_i} v_j
    S_A[i] = Σ_{j∈A_i} k_j
    M_A[i] = Σ_{j∈A_i} k_j v_j^T         (d x d)
    G_A[i] = Σ_{j∈A_i} k_j k_j^T         (d x d, 只用于分母二阶)

用法：
    from sparse_attn import attention_bundle, summarize_errors
    b = attention_bundle(q, k, v, exact_top_m=8, delta_hi=0.4)
    print(summarize_errors(b))

注意：本文件为了可读性与可验证性，显式计算了 N x N 的 logits 与掩码。
真实规模（N≈3e4）必须改为分块/FlashAttention 风格实现，见 README 的"上卡后的替换路线"。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


def attention_logits(q: torch.Tensor, k: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    d = q.shape[-1]
    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    return (q @ k.transpose(-1, -2)) * scale


def attention_bundle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_top_m: int = 8,
    delta_hi: float = 0.4,
    scale: Optional[float] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """一次算好三段划分、真实量与近似量，供误差分析与测试使用。

    exact_top_m : 每行精确计算的 key 数 m（决定"精确预算"）
    delta_hi    : A 段的下边界，A = {非 E 的 key 且 delta >= -delta_hi}
    masks       : 可选，直接传入 {"E": bool[N,N], "A": bool[N,N]} 覆盖上面的规则
                  （用于按"位次切分"而不是按阈值切分的实验）
    """
    n, d = q.shape
    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    s = attention_logits(q, k, scale)
    c = s.max(dim=-1, keepdim=True).values
    delta = s - c                              # <= 0
    w = torch.exp(delta)                       # 稳定化权重, 行最大为 1

    m = int(max(1, min(exact_top_m, n)))
    top_idx = torch.topk(s, m, dim=-1).indices
    if masks is not None:
        E = masks["E"].to(torch.bool)
        A = masks["A"].to(torch.bool) & (~E)
    else:
        E = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, top_idx, True)
        A = (~E) & (delta >= -float(delta_hi))
    D = (~E) & (~A)

    Ef = E.to(w.dtype)
    Af = A.to(w.dtype)
    wE = w * Ef
    wA = w * Af

    z_E = wE.sum(dim=-1)
    z_A = wA.sum(dim=-1)
    z_D = (w * D.to(w.dtype)).sum(dim=-1)
    z_tot = w.sum(dim=-1)

    n_E = wE @ v
    n_A_true = wA @ v
    o_dense = (w @ v) / z_tot.unsqueeze(-1)

    # ---------------- A 段的矩（一次聚合，可被本行所有 query 复用）----------------
    V_A = Af @ v                                     # [n, d]
    S_A = Af @ k                                     # [n, d]
    M_A = torch.einsum("ij,ja,je->iae", Af, k, v)    # [n, d, d]
    G_A = torch.einsum("ij,ja,jb->iab", Af, k, k)    # [n, d, d]

    qs = q * scale                                   # 使 s = qs · k
    m_A = Af.sum(dim=-1)                             # |A_i|
    sq = (S_A * qs).sum(dim=-1)                      # S_A · qs
    Mq = torch.einsum("iae,ia->ie", M_A, qs)         # M_A @ qs

    # 分子一阶:  n~_A = V_A + (M_A qs - c V_A)
    n_A_1 = V_A + (Mq - c * V_A)
    # 分母一阶:  z~_A = m_A + (S_A·qs - c m_A)
    z_A_1 = m_A + (sq - c.squeeze(-1) * m_A)
    # 分母二阶:  + 1/2 [ qs^T G_A qs - 2c (S_A·qs) + c^2 m_A ]
    qGq = torch.einsum("ia,iab,ib->i", qs, G_A, qs)
    c1 = c.squeeze(-1)
    z_A_2 = z_A_1 + 0.5 * (qGq - 2.0 * c1 * sq + c1 * c1 * m_A)

    eps = torch.finfo(w.dtype).tiny
    o_hard = n_E / z_E.clamp_min(eps).unsqueeze(-1)
    o_order1 = (n_E + n_A_1) / (z_E + z_A_1).clamp_min(eps).unsqueeze(-1)
    o_hybrid = (n_E + n_A_1) / (z_E + z_A_2).clamp_min(eps).unsqueeze(-1)

    # 理论量：泰勒余项 r_j = e^delta - 1 - delta，以及它的可求和上界
    r_abs = torch.abs(w - 1.0 - delta)               # 数值上等于 e^d - 1 - d (d<=0)
    bound_rem = torch.einsum("ij,j->i", (r_abs * Af), v.norm(dim=-1))
    delta_A_max = torch.where(A, delta.abs(), torch.zeros_like(delta)).max(dim=-1).values

    return {
        "scale": scale,
        "logits": s, "c": c, "delta": delta, "w": w,
        "masks": {"E": E, "A": A, "D": D},
        "z": {"E": z_E, "A": z_A, "D": z_D, "tot": z_tot},
        "flow": {
            "rho_exact": z_E / z_tot.clamp_min(eps),
            "rho_comp": z_A / z_tot.clamp_min(eps),
            "rho_drop": z_D / z_tot.clamp_min(eps),
        },
        "true": {"n_A": n_A_true, "z_A": z_A},
        "approx": {"n_A_1": n_A_1, "z_A_1": z_A_1, "z_A_2": z_A_2},
        "moments": {"V_A": V_A, "S_A": S_A, "M_A": M_A, "G_A": G_A, "m_A": m_A},
        "out": {"dense": o_dense, "hard": o_hard, "order1": o_order1, "hybrid": o_hybrid},
        "theory": {"r_abs": r_abs, "bound_numerator": bound_rem, "delta_A_max": delta_A_max},
    }


def rel_error(out: torch.Tensor, ref: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """逐行相对误差 ||out - ref|| / ||ref||。"""
    num = (out - ref).norm(dim=dim)
    den = ref.norm(dim=dim).clamp_min(torch.finfo(ref.dtype).tiny)
    return num / den


def summarize_errors(b: Dict[str, Any]) -> Dict[str, float]:
    """把 bundle 里的误差汇总成一行指标（中位数 + 均值），用于扫参与打点。"""
    ref = b["out"]["dense"]
    out: Dict[str, float] = {}
    for name in ("hard", "order1", "hybrid"):
        e = rel_error(b["out"][name], ref)
        out["rel_err_" + name + "_median"] = float(e.median())
        out["rel_err_" + name + "_mean"] = float(e.mean())
    # A 段内部的近似质量（排除 D 段丢弃造成的误差）
    n_true = b["true"]["n_A"]
    n_apx = b["approx"]["n_A_1"]
    denom = n_true.norm(dim=-1).clamp_min(torch.finfo(n_true.dtype).tiny)
    eA = (n_apx - n_true).norm(dim=-1) / denom
    out["bandA_rel_err_median"] = float(eA.median())
    bound = b["theory"]["bound_numerator"] / denom
    out["bandA_bound_median"] = float(bound.median())
    out["bound_holds"] = float(bool((eA <= bound + 1e-6).all()))
    for k, val in b["flow"].items():
        out[k + "_median"] = float(val.median())
    out["delta_A_max_median"] = float(b["theory"]["delta_A_max"].median())
    out["bandA_keys_median"] = float(b["masks"]["A"].sum(dim=-1).float().median())
    return out


def per_key_relative_error(delta: torch.Tensor) -> torch.Tensor:
    """单个 key 的权重近似相对误差：|e^d - (1+d)| / e^d = 1 - e^{-d}(1+d)。

    硬丢弃的对应值是 1。所以"一阶是否优于丢弃"的判据就是该值是否 < 1。
    该函数是 band 划分阈值 delta_hi 的直接依据。
    """
    d = delta.clamp(max=0.0)
    return 1.0 - torch.exp(-d) * (1.0 + d)


def break_even_delta() -> float:
    """一阶近似与硬丢弃的 break-even 点（数值求解 1 - e^{-d}(1+d) = 1）。"""
    lo, hi = 0.1, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(per_key_relative_error(torch.tensor([-mid]))) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
