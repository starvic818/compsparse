"""稀疏可行性分析：给定注意力 logits 的分布，回答"我能安全地稀疏到多少"。

这是无 GPU 也能做的核心分析，而且是上卡前最该先知道的事：
    稀疏率不是自由参数，它由该层/该头的 logits 分布决定。
    补偿（泰勒）只在"平坦注意力"区间有效；logit 越分散，A 段越窄，补偿收益越接近 0。

三种策略在同一 sparsity 下的对比（按位次切分，确保公平）：
    hard   : 只保留 top-m 精确，其余全部丢弃
    comp   : top-m 精确 + 紧随其后的若干 key 用一阶泰勒补偿（band 上限由 tau 控制）+ 剩下丢弃
    dense  : 参考

用法：
    python feasibility.py --synthetic --outdir results_feasibility
    # 上卡后拿到真实 logits 再跑：
    python feasibility.py --delta-file real_delta.npy --outdir results_feasibility_real

--delta-file 需要 shape [n_rows, N] 的 delta 矩阵（delta = s - rowmax，<= 0）。
可以用 capture_qkv.py 从 Wan2.1 的某一层导出（见 README）。
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sparse_attn import attention_bundle, rel_error


# ------------------------------------------------------------------ 合成数据

def toy_qkv(n: int = 256, d: int = 32, seed: int = 8, k_scale: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, d, generator=g)
    k = torch.randn(n, d, generator=g) * k_scale
    v = torch.randn(n, d, generator=g)
    return q, k, v


def delta_matrix(q: torch.Tensor, k: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """delta = s - rowmax，逐行 <= 0。"""
    d = q.shape[-1]
    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    s = (q @ k.transpose(-1, -2)) * scale
    return s - s.max(dim=-1, keepdim=True).values


# ------------------------------------------------------------------ 切分规则

def rank_masks(delta: torch.Tensor, m: int, c: int) -> Dict[str, torch.Tensor]:
    """按 logit 位次切分：前 m 个 -> E，接下来 (c-m) 个 -> A，剩余 -> 丢弃。"""
    n = delta.shape[-1]
    m = max(1, min(int(m), n))
    c = max(m, min(int(c), n))
    E = torch.zeros_like(delta, dtype=torch.bool)
    A = torch.zeros_like(delta, dtype=torch.bool)
    E.scatter_(-1, delta.topk(m, dim=-1).indices, True)
    rest = delta.masked_fill(E, -float("inf"))
    k_band = c - m
    if k_band > 0:
        A.scatter_(-1, rest.topk(k_band, dim=-1).indices, True)
    # 注意：这里"前 m 个"用的是 delta 的位次（等价于 logits 位次），A 段紧随其后，
    # 保证 band 内部 |delta| 单调递增 —— 这正是补偿误差可控的原因。
    A = A & (~E)
    return {"E": E, "A": A}


def choose_c(delta: torch.Tensor, m: int, tau: float) -> torch.Tensor:
    """在固定精确预算 m 下，选择最大的 c 使 band 内一阶近似的加权相对误差 <= tau。

    单 key 相对误差 rr(d) = 1 - e^{-d}(1+d) 随 |d| 单调递增，
    所以 band 的加权平均误差随 c 单调递增 -> 一旦超过 tau 就可以停止扩展 band。
    """
    w = torch.exp(delta)
    rr = 1.0 - torch.exp(-delta.clamp(max=0.0)) * (1.0 + delta.clamp(max=0.0))
    order = delta.argsort(dim=-1, descending=True)
    w_s = w.gather(-1, order)
    rr_s = rr.gather(-1, order)
    cw = torch.cumsum(w_s, dim=-1)
    cr = torch.cumsum(rr_s * w_s, dim=-1)          # 误差按质量加权
    n = delta.shape[-1]
    c_out = torch.full((delta.shape[0],), m, dtype=torch.long)
    for i in range(delta.shape[0]):
        base_w = cw[i, m - 1]
        base_r = cr[i, m - 1]
        best = m
        for c in range(m + 1, n + 1):
            den = cw[i, c - 1] - base_w
            if den <= 0:
                best = c
                continue
            num = cr[i, c - 1] - base_r
            if float(num / den) <= tau:
                best = c
            else:
                break
        c_out[i] = best
    return c_out


def sparsity_analysis(
    delta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsities: Sequence[float],
    tau: float = 0.2,
    scale: Optional[float] = None,
) -> List[Dict[str, float]]:
    """对每个目标稀疏率，比较 hard / comp 两种策略的实测输出误差与质量占比。"""
    n = delta.shape[-1]
    rows: List[Dict[str, float]] = []
    for s in sparsities:
        m = max(1, int(round((1.0 - s) * n)))
        c = choose_c(delta, m, tau)
        # hard: band 为空
        rows_ = torch.arange(delta.shape[0])
        masks_hard = {"E": torch.zeros_like(delta, dtype=torch.bool), "A": torch.zeros_like(delta, dtype=torch.bool)}
        masks_hard["E"].scatter_(-1, delta.topk(m, dim=-1).indices, True)

        # comp: 每个 row 的 c 不同，逐行构造
        E = torch.zeros_like(delta, dtype=torch.bool)
        A = torch.zeros_like(delta, dtype=torch.bool)
        E.scatter_(-1, delta.topk(m, dim=-1).indices, True)
        for i in range(delta.shape[0]):
            ci = int(c[i])
            if ci > m:
                rest = delta[i].masked_fill(E[i], -float("inf"))
                A[i].scatter_(-1, rest.topk(ci - m, dim=-1).indices, True)
        A = A & (~E)

        b_hard = attention_bundle(q, k, v, exact_top_m=m, scale=scale, masks=masks_hard)
        b_comp = attention_bundle(q, k, v, exact_top_m=m, scale=scale, masks={"E": E, "A": A})

        e_hard = rel_error(b_hard["out"]["hard"], b_hard["out"]["dense"])
        e_comp = rel_error(b_comp["out"]["order1"], b_comp["out"]["dense"])
        rows.append(
            {
                "sparsity": float(s),
                "exact_keys": float(m),
                "comp_keys_median": float((A.sum(-1)).float().median()),
                "drop_keys_median": float((~(E | A)).sum(-1).float().median()),
                "rho_comp_median": float(b_comp["flow"]["rho_comp"].median()),
                "rho_drop_median": float(b_comp["flow"]["rho_drop"].median()),
                "err_hard_median": float(e_hard.median()),
                "err_comp_median": float(e_comp.median()),
                "err_hard_p90": float(e_hard.quantile(0.9)),
                "err_comp_p90": float(e_comp.quantile(0.9)),
                "gain_x": float(e_hard.median() / max(e_comp.median(), 1e-12)),
            }
        )
    return rows


# ------------------------------------------------------------------ 绘图

def band_partition(delta: torch.Tensor, delta_1: float, eps_tol: float) -> Dict[str, torch.Tensor]:
    """按 delta 阈值切三段（"中段精确"方案）：
       A（补偿）: delta >= -delta_1        —— 靠近行最大值，质量最大，而一阶近似在这里最准
       D（丢弃）: 尾部质量 <= eps_tol 比例的那一段
       E（精确）: 夹在中间、既不能安全丢弃、一阶又不够准的那一段
    直觉：一阶近似在 delta=0 处是精确的（1+0 = e^0），误差随 |delta| 增大而增大；
    因此"最重的 key"恰恰是补偿最准的地方，把它们拿去精确计算是浪费。
    """
    n = delta.shape[-1]
    order = delta.argsort(dim=-1, descending=True)
    w = torch.exp(delta)
    w_s = w.gather(-1, order)
    cw = torch.cumsum(w_s, dim=-1)
    total = cw[:, -1:]
    ok = ((total - cw) <= eps_tol * total)          # 保留前 j+1 个时的尾部质量是否达标
    p = ok.float().argmax(dim=-1) + 1               # 最小可行保留数（单调，取第一个 True）
    p_A = (delta >= -float(delta_1)).sum(dim=-1)    # A 段 key 数
    rank = torch.empty_like(delta, dtype=torch.long)
    rank.scatter_(-1, order, torch.arange(n).expand_as(delta))
    A = rank < p_A
    p_keep = torch.maximum(p, p_A)
    D = rank >= p_keep.unsqueeze(-1)
    E = (~A) & (~D)
    return {"E": E, "A": A, "D": D}


def compare_partitions(
    delta: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta_1: float = 0.4,
    eps_tol: float = 0.02,
    scale: Optional[float] = None,
) -> List[Dict[str, float]]:
    """同一行内比较两种"精确预算分配"方案（稀疏率对齐到完全相同的精确 key 数）：
       top_m : 标准做法 —— 精确集 = logit 最高的 m 个
       band_E: 本推导的做法 —— 补偿近端高 logit、精确中段、丢弃尾部
    两者的精确 key 数相同（m_i = |E_i|），所以计算量可比。
    """
    n = delta.shape[-1]
    part = band_partition(delta, delta_1, eps_tol)
    m_per_row = part["E"].sum(dim=-1)

    b_band = attention_bundle(q, k, v, exact_top_m=1, scale=scale, masks={"E": part["E"], "A": part["A"]})
    e_band = rel_error(b_band["out"]["order1"], b_band["out"]["dense"])

    # 与 band 方案对齐精确预算的 top-m 方案
    E_top = torch.zeros_like(delta, dtype=torch.bool)
    for i in range(delta.shape[0]):
        mi = int(max(1, m_per_row[i]))
        E_top[i].scatter_(-1, delta[i].topk(mi, dim=-1).indices, True)
    A_top = torch.zeros_like(delta, dtype=torch.bool)
    for i in range(delta.shape[0]):
        mi = int(max(1, m_per_row[i]))
        ci = int(choose_c(delta[i : i + 1], mi, 0.2)[0])
        if ci > mi:
            rest = delta[i].masked_fill(E_top[i], -float("inf"))
            A_top[i].scatter_(-1, rest.topk(ci - mi, dim=-1).indices, True)
    A_top = A_top & (~E_top)
    b_top = attention_bundle(q, k, v, exact_top_m=1, scale=scale, masks={"E": E_top, "A": A_top})
    e_top = rel_error(b_top["out"]["order1"], b_top["out"]["dense"])

    return [
        {
            "strategy": "top_m_exact",
            "exact_keys_median": float(m_per_row.float().median()),
            "sparsity_median": float(1.0 - (m_per_row.float() / n).median()),
            "err_median": float(e_top.median()),
            "err_p90": float(e_top.quantile(0.9)),
        },
        {
            "strategy": "band_E_exact",
            "exact_keys_median": float(m_per_row.float().median()),
            "sparsity_median": float(1.0 - (m_per_row.float() / n).median()),
            "err_median": float(e_band.median()),
            "err_p90": float(e_band.quantile(0.9)),
        },
    ]


def delta_profile(
    delta: torch.Tensor,
    sparsities: Sequence[float],
    tau: float = 0.2,
    target_err: float = 0.02,
    max_scan: int = 200000,
    proxy: str = "worst",
) -> List[Dict[str, float]]:
    """只依赖 delta 的可行性画像（不需要 v），因此可以直接跑在真实规模上。

    与 sparsity_analysis 的区别：那个要算真实输出误差，需要 N×N 掩码与 einsum，只适合小 N；
    这个只用 logits 的分布，逐行 O(N)，可以在 N≈3e4 的真实张量上用。

    两个误差代理（都用 rho 归一，不含 v，因为真实规模下拿不到 v）：
      bound_worst = rho_drop + Σ_{j∈A} r_j w_j / z      （默认）
          —— 假设误差同向叠加。合成数据标定（`--calibrate` 可复现）：
             · 补偿主导区间（rho_drop ≈ 0）：比值 0.71~1.24，±30% 内，可直接当代理用；
             · 丢弃主导区间（rho_drop 大）：比值 0.32~0.65，**低估 1.5~3 倍**
               （因为真实误差还包含"精确段均值与尾部均值之差"，这项不在 rho 里）。
             所以 rho_drop 较大时本代理是乐观的，见输出列 `optimistic`。
      proxy_rms   = rho_drop + sqrt( Σ_{j∈A} (r_j w_j)^2 ) / z
          —— 假设 v_j 独立、误差按平方和叠加。实测低估约 10 倍，**不建议用于判定**，仅作参考。
    注意：本函数的 rho_drop 是【补偿方案】下的丢弃质量占比；若要对应"纯硬丢弃"，
    应使用 rho_comp + rho_drop（band 也会被丢掉）。两者都只是**排序/判定工具**，
    不等于最终的 FVD 与闪烁指标。

    返回每个稀疏率一行：computed keys / band keys / dropped keys / rho / 代理误差 / 是否达标。
    """
    n = delta.shape[-1]
    w = torch.exp(delta)
    rr = 1.0 - torch.exp(-delta.clamp(max=0.0)) * (1.0 + delta.clamp(max=0.0))
    order = delta.argsort(dim=-1, descending=True)
    w_s = w.gather(-1, order)
    rr_s = rr.gather(-1, order)
    d_s = delta.gather(-1, order)
    cw = torch.cumsum(w_s, dim=-1)
    cr = torch.cumsum(rr_s * w_s, dim=-1)
    total = cw[:, -1:].clamp_min(1e-12)

    rows: List[Dict[str, float]] = []
    for s in sparsities:
        m = max(1, int(round((1.0 - s) * n)))
        rho_drop = torch.ones(delta.shape[0])
        rho_comp = torch.zeros(delta.shape[0])
        band_keys = torch.zeros(delta.shape[0])
        band_dmax = torch.zeros(delta.shape[0])
        bound_worst = torch.zeros(delta.shape[0])
        rms = torch.zeros(delta.shape[0])
        for i in range(delta.shape[0]):
            base_w = cw[i, m - 1]
            base_r = cr[i, m - 1]
            c = m
            steps = min(n, m + max_scan)
            for cc in range(m + 1, steps + 1):
                den = cw[i, cc - 1] - base_w
                if float(den) <= 0:
                    c = cc
                    continue
                if float((cr[i, cc - 1] - base_r) / den) <= tau:
                    c = cc
                else:
                    break
            band_keys[i] = c - m
            if c > m:
                rho_comp[i] = float((cw[i, c - 1] - base_w) / total[i, 0])
                band_dmax[i] = float(-d_s[i, c - 1])
                rw = rr_s[i, m:c] * w_s[i, m:c]
                bound_worst[i] = float(rw.sum())
                rms[i] = float(torch.sqrt((rw ** 2).sum()))
            rho_drop[i] = float((total[i, 0] - cw[i, c - 1]) / total[i, 0])
        bound_norm = bound_worst / total[:, 0]
        rms_norm = rms / total[:, 0]
        chosen = rms_norm if proxy == "rms" else bound_norm
        proxy_err = rho_drop + chosen
        rows.append(
            {
                "sparsity": float(s),
                "exact_keys": float(m),
                "band_keys_median": float(band_keys.median()),
                "band_max_absdelta_median": float(band_dmax.median()),
                "rho_drop_median": float(rho_drop.median()),
                "rho_comp_median": float(rho_comp.median()),
                "bound_worst_median": float(bound_norm.median()),
                "proxy_rms_median": float(rms_norm.median()),
                "proxy_err_median": float(proxy_err.median()),
                "proxy_err_p90": float(proxy_err.quantile(0.9)),
                "safe_rows_frac": float((proxy_err <= target_err).float().mean()),
                "optimistic": bool(float(rho_drop.median()) > 0.2),
                "target_err": float(target_err),
            }
        )
    return rows


def max_safe_sparsity(
    delta: torch.Tensor,
    target_err: float = 0.02,
    tau: float = 0.2,
    grid: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """在给定误差目标下，这个（层, 头）最多能稀疏到多少。用于给层/头排序。"""
    grid = grid or [i / 100.0 for i in range(0, 100, 5)]
    rows = delta_profile(delta, grid, tau=tau, target_err=target_err)
    ok = [r for r in rows if r["safe_rows_frac"] >= 0.9]
    best = max((r["sparsity"] for r in ok), default=0.0)
    return {
        "max_safe_sparsity": float(best),
        "target_err": float(target_err),
        "median_proxy_at_best": float(next((r["proxy_err_median"] for r in rows if r["sparsity"] == best), float("nan"))),
    }

def plot_synthetic(results: Dict[float, List[Dict[str, float]]], out_png: str, tau: float) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=160)
    for ks, rows in sorted(results.items()):
        s = [r["sparsity"] for r in rows]
        eh = [r["err_hard_median"] for r in rows]
        ec = [r["err_comp_median"] for r in rows]
        axes[0].plot(s, eh, "--", marker="o", ms=3, label="k_scale=%.2f hard" % ks)
        axes[0].plot(s, ec, "-", marker="s", ms=3, label="k_scale=%.2f comp" % ks)
        axes[1].plot(s, [r["rho_drop_median"] for r in rows], "--", marker="o", ms=3, label="k=%.2f drop mass" % ks)
        axes[1].plot(s, [r["rho_comp_median"] for r in rows], "-", marker="s", ms=3, label="k=%.2f comp mass" % ks)

    axes[0].set_yscale("log")
    axes[0].set_title("output rel. error vs sparsity (tau=%.2f)" % tau, fontsize=9)
    axes[0].set_xlabel("sparsity (fraction of keys not computed exactly)")
    axes[0].set_ylabel("median rel. error (log)")
    axes[1].set_title("mass split: compensated vs dropped", fontsize=9)
    axes[1].set_xlabel("sparsity")
    axes[1].set_ylabel("fraction of attention mass")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------ CLI

def _torch_load(path: str):
    """兼容 torch 2.6+ 把 weights_only 默认改为 True 的变化。
    我们的抓取文件是纯容器 + 张量，优先用 weights_only=True（更安全），失败再回退。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:      # torch < 2.0 没有该参数
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="用合成 logits 做相图扫描")
    ap.add_argument("--compare", action="store_true",
                    help="额外比较两种精确预算分配方式（top-m vs 中段精确）")
    ap.add_argument("--calibrate", action="store_true",
                    help="在合成分布上标定误差代理：对比 bound_worst / proxy_rms 与真实误差")
    ap.add_argument("--delta-file", default=None, help="真实 delta 矩阵 .npy [n_rows, N]")
    ap.add_argument("--qkv-pt", default=None, help="真实 q/k/v 的 .pt（含 q,k,v 三个张量），用于算真实误差")
    ap.add_argument("--k-scales", default="0.02,0.05,0.1,0.2,0.35,0.6,1.0")
    ap.add_argument("--sparsities", default="0.25,0.5,0.7,0.8,0.9,0.95")
    ap.add_argument("--tau", type=float, default=0.2, help="band 内允许的加权相对误差上限")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--outdir", default="results_feasibility")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    sparsities = [float(x) for x in args.sparsities.split(",") if x.strip()]
    rows_all: List[Dict[str, float]] = []
    results: Dict[float, List[Dict[str, float]]] = {}

    if args.delta_file:
        delta = torch.from_numpy(np.load(args.delta_file)).float()
        if args.qkv_pt is None:
            raise SystemExit("--delta-file 模式还需要 --qkv-pt 才能算真实输出误差")
        pack = _torch_load(args.qkv_pt)
        rows = sparsity_analysis(delta, pack["q"], pack["k"], pack["v"], sparsities, tau=args.tau)
        for r in rows:
            r["source"] = "real"
        rows_all.extend(rows)
        results[float("nan")] = rows
    else:
        for ks in [float(x) for x in args.k_scales.split(",") if x.strip()]:
            q, k, v = toy_qkv(n=args.n, d=args.d, seed=args.seed, k_scale=ks)
            delta = delta_matrix(q, k)
            std = float(delta.std())
            rows = sparsity_analysis(delta, q, k, v, sparsities, tau=args.tau)
            for r in rows:
                r["k_scale"] = ks
                r["logit_std"] = std
                r["source"] = "synthetic"
            rows_all.extend(rows)
            results[ks] = rows

    import pandas as pd

    df = pd.DataFrame(rows_all)
    csv_path = os.path.join(args.outdir, "feasibility.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig_path = ""
    if results and not args.delta_file:
        fig_path = plot_synthetic(results, os.path.join(args.outdir, "feasibility.png"), args.tau)

    if args.compare:
        rows_cmp: List[Dict[str, float]] = []
        for ks in [float(x) for x in args.k_scales.split(",") if x.strip()]:
            q, k, v = toy_qkv(n=args.n, d=args.d, seed=args.seed, k_scale=ks)
            delta = delta_matrix(q, k)
            for r in compare_partitions(delta, q, k, v, delta_1=0.4, eps_tol=0.02):
                r["k_scale"] = ks
                r["logit_std"] = float(delta.std())
                rows_cmp.append(r)
        cmp_df = pd.DataFrame(rows_cmp)
        cmp_path = os.path.join(args.outdir, "partition_compare.csv")
        cmp_df.to_csv(cmp_path, index=False, encoding="utf-8-sig")
        print("\n--- 精确预算分配对照 (delta_1=0.4, eps_tol=2%) ---")
        print(cmp_df[["k_scale", "logit_std", "strategy", "exact_keys_median",
                      "sparsity_median", "err_median", "err_p90"]]
              .to_string(index=False, float_format=lambda x: "%.4g" % x))
        print("写出:", cmp_path)

    if args.calibrate:
        rows_cal: List[Dict[str, float]] = []
        for ks in [float(x) for x in args.k_scales.split(",") if x.strip()]:
            q, k, v = toy_qkv(n=args.n, d=args.d, seed=args.seed, k_scale=ks)
            delta = delta_matrix(q, k)
            exact = {r["sparsity"]: r for r in sparsity_analysis(delta, q, k, v, sparsities, tau=args.tau)}
            for r in delta_profile(delta, sparsities, tau=args.tau):
                e = exact.get(r["sparsity"])
                if e is None:
                    continue
                actual = max(e["err_comp_median"], 1e-12)
                rows_cal.append(
                    {
                        "k_scale": ks,
                        "logit_std": float(delta.std()),
                        "sparsity": r["sparsity"],
                        "rho_drop": r["rho_drop_median"],
                        "rho_comp": r["rho_comp_median"],
                        "bound_worst": r["bound_worst_median"],
                        "proxy_rms": r["proxy_rms_median"],
                        "actual_err_comp": e["err_comp_median"],
                        "actual_err_hard": e["err_hard_median"],
                        "proxy_worst_total": r["rho_drop_median"] + r["bound_worst_median"],
                        "proxy_rms_total": r["rho_drop_median"] + r["proxy_rms_median"],
                        "worst_over_actual": (r["rho_drop_median"] + r["bound_worst_median"]) / actual,
                        "rms_over_actual": (r["rho_drop_median"] + r["proxy_rms_median"]) / actual,
                    }
                )
        cal_df = pd.DataFrame(rows_cal)
        cal_path = os.path.join(args.outdir, "proxy_calibration.csv")
        cal_df.to_csv(cal_path, index=False, encoding="utf-8-sig")
        show = [c for c in ("logit_std", "sparsity", "rho_drop", "rho_comp", "bound_worst",
                            "proxy_rms", "actual_err_comp", "proxy_worst_total",
                            "worst_over_actual", "rms_over_actual") if c in cal_df.columns]
        print("\n--- 误差代理标定（bound_worst 应接近 1.0；rms 明显偏小则不可用）---")
        print(cal_df[show].to_string(index=False, float_format=lambda x: "%.4g" % x))
        print("写出:", cal_path)

    cols = [c for c in ("k_scale", "logit_std", "sparsity", "exact_keys", "comp_keys_median",
                        "rho_comp_median", "rho_drop_median", "err_hard_median",
                        "err_comp_median", "gain_x") if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda x: "%.4g" % x))
    print("\n写出:", csv_path, fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
