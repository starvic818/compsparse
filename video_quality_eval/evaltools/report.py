"""报告与图：把配对统计结果整理成可直接贴进实践报告的 markdown 与图。

核心图是"效率-质量前沿"：x = 相对 baseline 的加速比，y = 质量指标（闪烁指数等），
每个点 = 一个稀疏率配置，每条曲线 = 一种策略（硬丢弃 / 一阶补偿 / 混合阶）。
这张图比"某稀疏率下 2.1x 加速"有说服力得多，因为它同时展示了折中点与交叉趋势。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def df_to_markdown(df, floatfmt: str = "%.4g") -> str:
    """不依赖 tabulate 的简易 markdown 表格。"""
    if df is None or len(df) == 0:
        return "_（空表）_\n"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if v != v else floatfmt % v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 图内文字统一用 ASCII：服务器普遍缺少中文字体，中文会渲染成方框（tofu）。
    # 如需中文标签，请先安装字体（如 Noto Sans CJK）并设置 plt.rcParams["font.sans-serif"]。
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def frontier_plot(
    points: Sequence[Dict[str, Any]],
    out_png: str,
    x_key: str = "speedup_vs_baseline",
    y_key: str = "flicker_index",
    series_key: str = "strategy",
    title: str = "Efficiency-quality frontier (x: speedup, y: flicker_index)",
) -> str:
    plt = _plt()
    series: Dict[str, List[Dict[str, Any]]] = {}
    for p in points:
        if not np.isfinite(p.get(x_key, float("nan"))) or not np.isfinite(p.get(y_key, float("nan"))):
            continue
        series.setdefault(str(p.get(series_key, "default")), []).append(p)

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=160)
    for name, pts in sorted(series.items()):
        pts = sorted(pts, key=lambda r: r[x_key])
        xs = [r[x_key] for r in pts]
        ys = [r[y_key] for r in pts]
        ax.plot(xs, ys, marker="o", label=name)
        for r in pts:
            ax.annotate(str(r.get("label", "")), (r[x_key], r[y_key]), fontsize=6, alpha=0.7)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("speedup vs baseline (x)")
    ax.set_ylabel(y_key)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def paired_delta_plot(paired_rows: Sequence[Dict[str, Any]], out_png: str, metric: Optional[str] = None) -> str:
    """每个 (tag, metric) 的配对 delta 与 95% CI。"""
    plt = _plt()
    rows = [r for r in paired_rows if metric is None or r["metric"] == metric]
    if not rows:
        return ""
    labels = ["%s\n%s" % (r["tag"], r["metric"]) for r in rows]
    ypos = np.arange(len(rows))
    deltas = np.array([r["delta_mean"] for r in rows], dtype=float)
    lo = np.array([r["ci_low"] for r in rows], dtype=float)
    hi = np.array([r["ci_high"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(rows) + 1.6), dpi=160)
    ax.errorbar(deltas, ypos, xerr=[deltas - lo, hi - deltas], fmt="o", capsize=3, lw=1.0)
    ax.axvline(0.0, color="gray", lw=0.9, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("paired delta = method - baseline (95% CI)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def flicker_vs_dropped_mass_plot(points: Sequence[Dict[str, Any]], out_png: str) -> str:
    """横轴 = 每行被近似/丢弃的注意力质量比例 rho，纵轴 = 闪烁指数。

    推导文档给出的预测是：闪烁 ∝ rho × 补偿项相对误差，理论上应近似线性。
    这张图是这个预测的直接检验。
    """
    plt = _plt()
    pts = [p for p in points if np.isfinite(p.get("rho_mean", float("nan"))) and np.isfinite(p.get("flicker_index", float("nan")))]
    if not pts:
        return ""
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=160)
    by = {}
    for p in pts:
        by.setdefault(str(p.get("strategy", "default")), []).append(p)
    for name, group in sorted(by.items()):
        group = sorted(group, key=lambda r: r["rho_mean"])
        ax.plot([g["rho_mean"] for g in group], [g["flicker_index"] for g in group], marker="s", label=name)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("mean dropped-mass ratio rho")
    ax.set_ylabel("flicker_index")
    ax.grid(alpha=0.3)
    if len(by) > 1:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def write_report(
    out_path: str,
    verify_problems: Sequence[str],
    video_info_rows,
    paired_rows: Sequence[Dict[str, Any]],
    speedup_rows: Sequence[Dict[str, Any]],
    figures: Dict[str, str],
    extra_notes: Optional[Sequence[str]] = None,
) -> str:
    """生成报告 markdown。"""
    import pandas as pd

    lines: List[str] = []
    lines.append("# 视频生成加速方案：配对质量-效率评测报告\n")
    lines.append("本报告由 `run_eval.py` 自动生成。所有质量结论都是按 (prompt_id, seed) 配对后的结果。\n")

    lines.append("\n## 1. 视频完整性校验\n")
    if verify_problems:
        lines.append("**存在问题的文件：**\n")
        for p in verify_problems:
            lines.append("- %s" % p)
    else:
        lines.append("全部视频通过校验（存在、可解码、帧数/分辨率/fps 符合预期、无内容重复）。\n")
    if video_info_rows:
        df = pd.DataFrame(video_info_rows)
        cols = [c for c in ("prompt_id", "tag", "n_frames", "width", "height", "fps", "nbytes", "mtime", "sha256_16") if c in df.columns]
        lines.append("\n")
        lines.append(df_to_markdown(df[cols]))

    lines.append("\n## 2. 加速比（实测）\n")
    if speedup_rows:
        lines.append(df_to_markdown(pd.DataFrame(list(speedup_rows))))
    else:
        lines.append("_未提供 timing 数据（--timing）_\n")

    lines.append("\n## 3. 配对质量对比\n")
    if paired_rows:
        df = pd.DataFrame(list(paired_rows))
        cols = [c for c in ("tag", "metric", "n_pairs", "base_mean", "method_mean", "delta_mean", "rel_delta_mean", "ci_low", "ci_high", "p_value", "cohen_dz", "verdict") if c in df.columns]
        lines.append(df_to_markdown(df[cols]))
    else:
        lines.append("_无配对结果_\n")

    lines.append("\n## 4. 图\n")
    for name, path in figures.items():
        if path:
            lines.append("### %s\n" % name)
            lines.append("![%s](%s)\n" % (name, os.path.basename(path)))

    if extra_notes:
        lines.append("\n## 5. 备注\n")
        for n in extra_notes:
            lines.append("- %s" % n)

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path
