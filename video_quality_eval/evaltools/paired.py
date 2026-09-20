"""配对统计：所有质量与速度结论都按 (prompt_id, seed) 配对后再做区间估计与检验。

为什么必须配对：同一 prompt、同一 seed 下，baseline 与稀疏方法的差异才是"方法造成的"。
跨 prompt 聚合会引入巨大的 prompt 难度方差，把 20% 的质量差异淹没在噪声里。

输出：
  delta = method - baseline（保留原始符号，不做正负号翻转）
  ci_low / ci_high : 对配对 delta 做 bootstrap 得到的 95% 置信区间
  p_value          : Wilcoxon 符号秩检验（scipy 缺失时退化为符号检验）
  cohen_dz         : 配对效应量
  verdict          : 三分类（改善 / 无显著差异 / 退化），由 CI 与容差共同决定
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """对样本均值的 bootstrap 置信区间（对配对 delta 使用）。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def _wilcoxon_p(base: np.ndarray, method: np.ndarray) -> float:
    diff = method - base
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    if np.allclose(diff, 0):
        return 1.0
    if _stats is not None:
        try:
            return float(_stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            pass
    # 退化：符号检验
    n_pos = int(np.sum(diff > 0))
    n = int(np.sum(diff != 0))
    if n == 0:
        return 1.0
    k = min(n_pos, n - n_pos)
    if _stats is not None:
        return float(min(1.0, 2 * _stats.binom.cdf(k, n, 0.5)))
    return float("nan")


def paired_stats(
    base: Sequence[float],
    method: Sequence[float],
    lower_is_better: bool = True,
    n_boot: int = 10000,
    tol_rel: float = 0.02,
    seed: int = 0,
) -> Dict[str, Any]:
    """对一组配对观测做完整统计。tol_rel 是"可接受劣化"的相对容差（默认 2%）。"""
    b = np.asarray(base, dtype=np.float64)
    m = np.asarray(method, dtype=np.float64)
    ok = np.isfinite(b) & np.isfinite(m)
    b, m = b[ok], m[ok]
    n = int(b.size)
    if n == 0:
        return {"n_pairs": 0, "verdict": "no_data"}

    delta = m - b
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(b) > 1e-12, delta / np.abs(b), np.nan)
    ci_low, ci_high = bootstrap_ci(delta, n_boot=n_boot, seed=seed)
    p = _wilcoxon_p(b, m)
    sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    dz = float(delta.mean() / sd) if sd > 1e-12 else 0.0

    base_mean = float(b.mean())
    tol_abs = tol_rel * abs(base_mean)
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        verdict = "insufficient"
    elif lower_is_better:
        if ci_high < -tol_abs:
            verdict = "improved"
        elif ci_low > tol_abs:
            verdict = "degraded"
        else:
            verdict = "no_significant_difference"
    else:
        if ci_low > tol_abs:
            verdict = "improved"
        elif ci_high < -tol_abs:
            verdict = "degraded"
        else:
            verdict = "no_significant_difference"

    return {
        "n_pairs": n,
        "base_mean": base_mean,
        "method_mean": float(m.mean()),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "rel_delta_mean": float(np.nanmean(rel)) if np.isfinite(rel).any() else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p,
        "cohen_dz": dz,
        "verdict": verdict,
    }


def paired_table(
    df,
    metric_cols: Sequence[str],
    baseline_tag: str,
    tag_col: str = "tag",
    key_cols: Sequence[str] = ("prompt_id", "seed"),
    lower_is_better: Optional[Dict[str, bool]] = None,
    tol_rel: float = 0.02,
    n_boot: int = 10000,
):
    """把长表 DataFrame 转成配对统计表。

    df 需要至少包含 key_cols + [tag_col] + metric_cols；同一 (key, tag) 应唯一。
    返回 rows（list[dict]），每行 = 一个 (tag, metric) 的统计结果。
    """
    import pandas as pd

    lower_is_better = lower_is_better or {}
    rows: List[Dict[str, Any]] = []
    keys = list(key_cols)

    for metric in metric_cols:
        if metric not in df.columns:
            continue
        sub = df[keys + [tag_col, metric]].dropna(subset=[metric])
        wide = sub.pivot_table(index=keys, columns=tag_col, values=metric, aggfunc="mean")
        if baseline_tag not in wide.columns:
            continue
        for tag in [c for c in wide.columns if c != baseline_tag]:
            pair = wide[[baseline_tag, tag]].dropna()
            if pair.empty:
                continue
            st = paired_stats(
                pair[baseline_tag].values,
                pair[tag].values,
                lower_is_better=bool(lower_is_better.get(metric, True)),
                tol_rel=tol_rel,
                n_boot=n_boot,
            )
            st.update({"tag": tag, "metric": metric, "baseline_tag": baseline_tag})
            rows.append(st)
    return rows


def speedup_table(
    df_timing,
    baseline_tag: str,
    tag_col: str = "tag",
    latency_col: str = "latency_s",
    group_cols: Sequence[str] = (),
    sparsity_col: str = "sparsity",
):
    """按 tag（可选再按 group）汇总实测时延与加速比。"""
    import pandas as pd

    if df_timing is None or len(df_timing) == 0:
        return []
    cols = list(group_cols) + [tag_col, latency_col]
    if sparsity_col in df_timing.columns and sparsity_col not in cols:
        cols.append(sparsity_col)
    if "peak_mem_gb" in df_timing.columns:
        cols.append("peak_mem_gb")
    sub = df_timing[cols].dropna(subset=[latency_col])
    keys = list(group_cols) + [tag_col]
    agg_map = {latency_col: ["mean", "std", "count"]}
    if "peak_mem_gb" in sub.columns:
        agg_map["peak_mem_gb"] = ["mean", "max"]
    g = sub.groupby(keys).agg(agg_map)
    g.columns = ["_".join([c for c in col if c]) for col in g.columns]
    g = g.reset_index()

    out: List[Dict[str, Any]] = []
    for key_vals, chunk in g.groupby(list(group_cols)) if group_cols else [(None, g)]:
        base_rows = chunk[chunk[tag_col] == baseline_tag]
        base_lat = float(base_rows[latency_col + "_mean"].mean()) if len(base_rows) else float("nan")
        for _, r in chunk.iterrows():
            lat = float(r[latency_col + "_mean"])
            rec = {
                "group": key_vals if isinstance(key_vals, str) else (key_vals[0] if group_cols else "all"),
                "tag": r[tag_col],
                "latency_s_mean": lat,
                "latency_s_std": float(r.get(latency_col + "_std", float("nan"))),
                "n": int(r.get(latency_col + "_count", 0)),
                "speedup_vs_baseline": (base_lat / lat) if lat > 0 else float("nan"),
            }
            for extra in ("peak_mem_gb_mean", "peak_mem_gb_max", "sparsity"):
                if extra in r.index:
                    rec[extra] = float(r[extra])
            out.append(rec)
    return out


def quality_budget(
    paired_rows: Sequence[Dict[str, Any]],
    critical_metrics: Sequence[str],
    tol_rel: float = 0.05,
) -> List[str]:
    """按"关键指标不得显著退化"的规则做一次通关判定，返回警告文本列表。"""
    warnings_: List[str] = []
    for row in paired_rows:
        if row["metric"] not in critical_metrics:
            continue
        if row["verdict"] == "degraded":
            warnings_.append(
                "[不通过] %s 在 %s 上显著退化: delta=%.4g (95%% CI [%.4g, %.4g], p=%.3g)"
                % (row["metric"], row["tag"], row["delta_mean"], row["ci_low"], row["ci_high"], row["p_value"])
            )
        elif row["verdict"] == "no_significant_difference":
            warnings_.append(
                "[需注意] %s 在 %s 上与 baseline 无显著差异，但 CI 宽度较大([%.4g, %.4g])，"
                "说明样本量/方差还不足以支撑「无损」结论"
                % (row["metric"], row["tag"], row["ci_low"], row["ci_high"])
            )
    return warnings_
