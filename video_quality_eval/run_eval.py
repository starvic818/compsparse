"""一键评测入口：manifest -> 完整性校验 -> 指标 -> 配对统计 -> 报告与图。

典型用法（在云平台的项目根目录下）：

  # 1) 只做完整性校验 + 闪烁指标 + 配对统计（最快，调试期用）
  python run_eval.py --manifest manifest.json --outdir results_reflect \\
      --metrics verify,flicker

  # 2) 加官方 VBench 三个维度（需要先装 VBench 并指定 vbench_full_info.json）
  python run_eval.py --manifest manifest.json --outdir results_full \\
      --metrics verify,flicker,vbench,proxy \\
      --vbench-full-info /workspace/VBench/vbench_full_info.json

  # 3) 补充相对 FVD（样本量 >= 32 再跑）
  python run_eval.py --manifest manifest.json --outdir results_full \\
      --metrics fvd --fvd-extractor r3d

输出目录内容：
  video_info.csv        每个视频的探测信息（帧数/分辨率/hash/修改时间）
  verify_report.md      完整性校验结果
  metrics_wide.csv      每个 run 一行，各指标一列
  metrics_long.csv      长表，便于画图与统计
  paired_summary.csv    配对统计（delta / CI / p / 判定）
  speedup.csv           实测加速比
  report.md             汇总报告
  figs/*.png            效率-质量前沿、配对 delta、闪烁 vs 丢弃质量
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaltools import flicker as flicker_mod  # noqa: E402
from evaltools import paired as paired_mod  # noqa: E402
from evaltools import report as report_mod  # noqa: E402
from evaltools import videoutil  # noqa: E402

# 指标方向：True = 越小越好
LOWER_IS_BETTER = {
    "bg_l1_mean": True,
    "bg_l1_std": True,
    "bg_l1_p95": True,
    "warp_resid_mean": True,
    "warp_resid_std": True,
    "warp_resid_p95": True,
    "patch_dc_var_mean": True,
    "patch_dc_var_median": True,
    "patch_dc_var_p90": True,
    "patch_dc_lowfreq_ratio": True,
    "flicker_index": True,
    "lpips_bg_mean": True,
    "clip_consistency_proxy": False,
    "dinov2_consistency_proxy": False,
    "subject_consistency": False,
    "background_consistency": False,
    "motion_smoothness": False,
    "imaging_quality": False,
    "aesthetic_quality": False,
    "dynamic_degree": False,
    "fvd": True,
    "kid": True,
}

# 关键指标：任一显著退化即判定方案不达标（可改）
CRITICAL_METRICS = ("flicker_index", "background_consistency", "subject_consistency", "motion_smoothness")

# 从 manifest 原样透传到结果表的字段（用于加速比表与 rho-闪烁 线性检验）
PASSTHROUGH = ("prompt_id", "group", "seed", "tag", "sparsity", "strategy",
               "latency_s", "peak_mem_gb", "rho_mean")


# ------------------------------------------------------------------ manifest

def load_manifest(path: str) -> Dict[str, Any]:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            data = {"runs": data}
        return data
    if path.lower().endswith(".csv"):
        import pandas as pd

        return {"runs": pd.read_csv(path).to_dict("records")}
    raise ValueError("manifest 只支持 .json / .csv")


def _baseline_index(runs: Sequence[Dict[str, Any]], baseline_tag: str) -> Dict[Any, str]:
    """(prompt_id, seed) -> baseline 视频路径。"""
    idx = {}
    for r in runs:
        if r.get("tag") == baseline_tag:
            key = (str(r.get("prompt_id")), str(r.get("seed")))
            if key in idx:
                raise ValueError("baseline 重复: %s" % (key,))
            idx[key] = r["video"]
    return idx


def _long_rows(records: Sequence[Dict[str, Any]], metric_cols: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for rec in records:
        for m in metric_cols:
            if m in rec and rec[m] is not None:
                rows.append(
                    {
                        "prompt_id": rec.get("prompt_id"),
                        "group": rec.get("group", "all"),
                        "seed": rec.get("seed"),
                        "tag": rec.get("tag"),
                        "sparsity": rec.get("sparsity", 0.0),
                        "strategy": rec.get("strategy", rec.get("tag")),
                        "metric": m,
                        "value": float(rec[m]),
                    }
                )
    return rows


# ------------------------------------------------------------------ 主流程

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="视频加速方案配对质量-效率评测")
    p.add_argument("--manifest", required=True)
    p.add_argument("--outdir", default="results")
    p.add_argument("--baseline-tag", default=None, help="默认取 manifest 里的 baseline_tag，再退回 'dense'")
    p.add_argument("--metrics", default="verify,flicker", help="逗号分隔: verify,flicker,proxy,vbench,fvd")
    p.add_argument("--max-side", type=int, default=512, help="闪烁指标计算时缩放到的长边（0=不缩放）")
    p.add_argument("--max-frames", type=int, default=0, help="只取前 N 帧（0=全部）")
    p.add_argument("--patch-grid", type=int, default=4)
    p.add_argument("--lowpass-sigma", type=float, default=2.0)
    p.add_argument("--mask-percentile", type=float, default=30.0)
    p.add_argument("--tol-rel", type=float, default=0.02, help="判定「显著退化」的相对容差")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--timing", default=None, help="可选：timing.csv，含 prompt_id/tag/latency_s/peak_mem_gb")
    p.add_argument("--with-lpips", action="store_true")
    p.add_argument("--proxy-backend", default="dinov2", choices=["dinov2", "clip"])
    p.add_argument("--vbench-full-info", default=None)
    p.add_argument("--vbench-device", default="cuda")
    p.add_argument("--vbench-dims", default="subject_consistency,background_consistency,motion_smoothness")
    p.add_argument("--vbench-cmd", default=None)
    p.add_argument("--vbench-name", default="evalrun")
    p.add_argument("--fvd-extractor", default="r3d", choices=["r3d", "common_metrics"])
    args = p.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    figs_dir = os.path.join(args.outdir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    manifest = load_manifest(args.manifest)
    runs: List[Dict[str, Any]] = manifest["runs"]
    expected = manifest.get("expected", {})
    baseline_tag = args.baseline_tag or manifest.get("baseline_tag", "dense")
    run_started_ts = manifest.get("started_ts")
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    max_side = None if args.max_side in (0, -1) else args.max_side
    max_frames = None if args.max_frames in (0, -1) else args.max_frames

    print("[1/6] 载入 manifest: %d 个 run, baseline_tag=%s" % (len(runs), baseline_tag))

    # ---------------------------------------------------------- 完整性校验
    problems: List[str] = []
    video_rows: List[Dict[str, Any]] = []
    if "verify" in metrics or "flicker" in metrics:
        print("[2/6] 校验视频文件 ...")
        problems, infos = videoutil.verify_manifest(runs, expected=expected, run_started_ts=run_started_ts)
        for run, info in zip(runs, infos):
            video_rows.append(
                {
                    "prompt_id": run.get("prompt_id"),
                    "tag": run.get("tag"),
                    "n_frames": info.get("n_frames"),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "fps": info.get("fps"),
                    "nbytes": info.get("nbytes"),
                    "mtime": info.get("mtime"),
                    "sha256_16": info.get("sha256_16"),
                }
            )
        if problems:
            print("  发现 %d 个问题，详见 %s/verify_report.md" % (len(problems), args.outdir))
            for pr in problems[:10]:
                print("   - %s" % pr)
        else:
            print("  全部通过。")
        with open(os.path.join(args.outdir, "verify_report.md"), "w", encoding="utf-8") as f:
            f.write("# 视频完整性校验\n\n")
            f.write("基准时间戳 started_ts=%s，期望 %s\n\n" % (run_started_ts, expected))
            if problems:
                for pr in problems:
                    f.write("- %s\n" % pr)
            else:
                f.write("全部通过。\n")

    # ---------------------------------------------------------- 闪烁指标
    baseline_videos = _baseline_index(runs, baseline_tag)
    records: List[Dict[str, Any]] = []
    if "flicker" in metrics:
        print("[3/6] 计算闪烁指标（背景掩码来自配对的 baseline 视频）...")
        for i, run in enumerate(runs):
            key = (str(run.get("prompt_id")), str(run.get("seed")))
            ref = baseline_videos.get(key)
            if run.get("tag") == baseline_tag:
                ref = None
            elif ref is None:
                print("   [warn] 找不到配对 baseline，掩码退化为自身: %s" % (key,))
            t0 = time.time()
            prof = flicker_mod.flicker_from_files(
                run["video"],
                ref_video_path=ref,
                max_frames=max_frames,
                max_side=max_side,
                mask_percentile=args.mask_percentile,
                patch_grid=args.patch_grid,
                lowpass_sigma=args.lowpass_sigma,
                with_lpips=args.with_lpips,
            )
            if ref is None:
                # baseline 行按定义与自己比：ratio=1.0, flicker_index=1.0。
                # 这样配对统计里 method/base 两侧都有定义（注意：baseline 的 1.0 是定义值，
                # 不是独立测量值，报告中需说明）。
                prof["flicker_index"] = 1.0
                for k in ("bg_l1", "warp_resid", "patch_dc_var"):
                    prof[k + "_ratio"] = 1.0
            prof.update({k: run.get(k) for k in PASSTHROUGH})
            records.append(prof)
            print("   [%d/%d] %s %s: flicker_index=%.3f (%.1fs)"
                  % (i + 1, len(runs), run.get("prompt_id"), run.get("tag"),
                     prof.get("flicker_index", float("nan")), time.time() - t0))
    else:
        records = [{k: run.get(k) for k in PASSTHROUGH} for run in runs]

    # ---------------------------------------------------------- 代理一致性
    if "proxy" in metrics:
        print("[4/6] 计算代理一致性（非官方 VBench）...")
        from evaltools import proxy_consistency as proxy_mod

        for rec, run in zip(records, runs):
            key = (str(run.get("prompt_id")), str(run.get("seed")))
            ref = baseline_videos.get(key)
            try:
                out = proxy_mod.proxy_consistency(
                    run["video"],
                    backend=args.proxy_backend,
                    background_only=False,
                    ref_video_path=ref,
                    max_frames=max_frames,
                )
                rec.update({k: v for k, v in out.items() if isinstance(v, (int, float))})
            except Exception as exc:  # noqa: BLE001
                print("   [warn] proxy 失败 %s: %s" % (run.get("prompt_id"), exc))
    elif "vbench" not in metrics:
        print("[4/6] 跳过代理一致性")

    # ---------------------------------------------------------- VBench
    vbench_note: List[str] = []
    if "vbench" in metrics:
        print("[5/6] 调用官方 VBench ...")
        from evaltools import vbench_adapter as vb

        if not args.vbench_full_info:
            vbench_note.append("未提供 --vbench-full-info，跳过 VBench（需要 VBench/vbench_full_info.json）")
        else:
            prompts = sorted({str(r["prompt_id"]) for r in runs})
            prompt_index = {pid: i for i, pid in enumerate(prompts)}
            idx_to_pid = {v: k for k, v in prompt_index.items()}
            dims = [d.strip() for d in args.vbench_dims.split(",") if d.strip()]
            for tag in sorted({r.get("tag") for r in runs}):
                subset = [r for r in runs if r.get("tag") == tag]
                # 每个 (tag, seed) 单独跑一次，保证"一个 prompt 一个视频"，
                # 这样 VBench 结果里的第 i 项就能可靠地映射回第 i 个 prompt。
                for seed in sorted({str(r.get("seed")) for r in subset}):
                    group = [r for r in subset if str(r.get("seed")) == seed]
                    run_name = "%s_%s_s%s" % (args.vbench_name, tag, seed)
                    layout = vb.build_vbench_layout(
                        group,
                        os.path.join(args.outdir, "vbench_input", str(tag), "seed%s" % seed),
                        prompt_index,
                    )
                    res = vb.run_vbench(
                        videos_dir=layout["dir"],
                        output_path=os.path.join(args.outdir, "vbench_out", str(tag), "seed%s" % seed),
                        name=run_name,
                        full_info_dir=args.vbench_full_info,
                        dimension_list=dims,
                        device=args.vbench_device,
                        cli_cmd_template=args.vbench_cmd,
                        use_cli=bool(args.vbench_cmd),
                    )
                    vbench_note.append(
                        "VBench[%s/s%s]: mode=%s, json=%s" % (tag, seed, res.get("mode"), res.get("results_json"))
                    )
                    if not res.get("results_json"):
                        vbench_note.append("  [warn] 未找到结果 JSON，请检查 VBench 版本与日志: %s" % res.get("stderr_tail", ""))
                        continue
                    try:
                        rows = vb.parse_vbench_results(res["results_json"])
                    except Exception as exc:  # noqa: BLE001
                        vbench_note.append("  [warn] 解析 VBench 结果失败: %s" % exc)
                        continue
                    n_expected = len({str(r["prompt_id"]) for r in group})
                    n_got = len({r["prompt_index"] for r in rows})
                    if n_got != n_expected:
                        vbench_note.append(
                            "  [warn] VBench 返回 %d 个视频结果，期望 %d 个 -> 跳过回填（索引可能错位）"
                            % (n_got, n_expected)
                        )
                        continue
                    by_key = {(r["dimension"], r["prompt_index"]): r["score"] for r in rows}
                    means = vb.dimension_means(rows)
                    for rec in records:
                        if rec.get("tag") != tag or str(rec.get("seed")) != seed:
                            continue
                        pid = idx_to_pid.get(prompt_index[str(rec.get("prompt_id"))])
                        for dim in dims:
                            key = (dim, prompt_index.get(str(pid)))
                            if key in by_key:
                                rec[dim] = by_key[key]
                    vbench_note.append("  %s/seed%s 维度均值: %s" % (tag, seed, json.dumps(means, ensure_ascii=False)))
    else:
        print("[5/6] 跳过 VBench")

    # ---------------------------------------------------------- FVD
    if "fvd" in metrics:
        print("[6a/6] 计算相对 FVD（小规模补充）...")
        from evaltools import fvd as fvd_mod

        try:
            ext = (
                fvd_mod.make_extractor_common_metrics(device=args.vbench_device)
                if args.fvd_extractor == "common_metrics"
                else fvd_mod.make_extractor_r3d(device=args.vbench_device)
            )
            base_paths = [baseline_videos[(str(r["prompt_id"]), str(r["seed"]))] for r in runs if r.get("tag") != baseline_tag]
            for tag in sorted({r.get("tag") for r in runs if r.get("tag") != baseline_tag}):
                tag_paths = [r["video"] for r in runs if r.get("tag") == tag]
                n = min(len(base_paths), len(tag_paths))
                if n == 0:
                    continue
                res = fvd_mod.bootstrap_fvd_ci(base_paths[:n], tag_paths[:n], extractor=ext, n_boot=100, device=args.vbench_device)
                vbench_note.append(
                    "相对 FVD[%s vs %s]: %.1f (95%% CI [%.1f, %.1f]), extractor=%s, n=%d"
                    % (tag, baseline_tag, res["fvd_point"], res["fvd_ci_low"], res["fvd_ci_high"],
                       getattr(ext, "name", "custom"), n)
                )
                if n < 32:
                    vbench_note.append("  [warn] n=%d < 32，FVD 置信区间会非常宽，只能当趋势参考" % n)
        except Exception as exc:  # noqa: BLE001
            vbench_note.append("FVD 计算失败: %s" % exc)
    else:
        print("[6a/6] 跳过 FVD")

    # ---------------------------------------------------------- 指标与统计
    print("[6b/6] 配对统计与报告 ...")
    metric_cols = sorted({k for rec in records for k in rec.keys() if k in LOWER_IS_BETTER})
    long_rows = _long_rows(records, metric_cols)

    import pandas as pd

    wide = pd.DataFrame(records)
    wide_path = os.path.join(args.outdir, "metrics_wide.csv")
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    long_df = pd.DataFrame(long_rows)
    long_path = os.path.join(args.outdir, "metrics_long.csv")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    paired_rows: List[Dict[str, Any]] = []
    if len(wide) and baseline_tag in set(wide.get("tag", [])):
        paired_rows = paired_mod.paired_table(
            wide,
            metric_cols,
            baseline_tag=baseline_tag,
            lower_is_better=LOWER_IS_BETTER,
            tol_rel=args.tol_rel,
            n_boot=args.n_boot,
        )
        if paired_rows:
            pd.DataFrame(paired_rows).to_csv(
                os.path.join(args.outdir, "paired_summary.csv"), index=False, encoding="utf-8-sig"
            )
        else:
            print("   [warn] 生成了 0 条配对结果（检查 baseline 与各方法是否共享同一 (prompt_id, seed)）")
    else:
        print("   [warn] 没有 baseline_tag=%s 的数据，跳过配对统计" % baseline_tag)

    timing_df = None
    if args.timing and os.path.isfile(args.timing):
        timing_df = pd.read_csv(args.timing)
    elif "latency_s" in wide.columns:
        timing_df = wide
    speedup_rows: List[Dict[str, Any]] = []
    if timing_df is not None and "latency_s" in timing_df.columns:
        group_cols = ["group"] if "group" in timing_df.columns else []
        speedup_rows = paired_mod.speedup_table(timing_df, baseline_tag=baseline_tag, group_cols=group_cols)
        if speedup_rows:
            pd.DataFrame(speedup_rows).to_csv(
                os.path.join(args.outdir, "speedup.csv"), index=False, encoding="utf-8-sig"
            )

    figures: Dict[str, str] = {}
    if speedup_rows and paired_rows:
        by_tag = {}
        for s in speedup_rows:
            by_tag[str(s["tag"])] = s
        points = []
        for r in paired_rows:
            if r["metric"] != "flicker_index":
                continue
            sp = by_tag.get(str(r["tag"]))
            if not sp:
                continue
            points.append(
                {
                    "speedup_vs_baseline": sp["speedup_vs_baseline"],
                    "flicker_index": r["method_mean"] / r["base_mean"] if r["base_mean"] else float("nan"),
                    "strategy": str(r["tag"]),
                    "label": "%.2fx" % sp.get("speedup_vs_baseline", float("nan")),
                }
            )
        if points:
            figures["效率-质量前沿"] = report_mod.frontier_plot(
                points, os.path.join(figs_dir, "frontier.png")
            )
    if paired_rows:
        figures["配对 delta（含 95% CI）"] = report_mod.paired_delta_plot(
            [r for r in paired_rows if r["metric"] in ("flicker_index", "bg_l1_mean", "patch_dc_var_mean", "background_consistency")],
            os.path.join(figs_dir, "paired_delta.png"),
        )

    notes = list(vbench_note)
    notes.append("判定容差 tol_rel=%.3f；关键指标: %s" % (args.tol_rel, ", ".join(CRITICAL_METRICS)))
    warnings_ = paired_mod.quality_budget(paired_rows, CRITICAL_METRICS, tol_rel=args.tol_rel)
    notes.extend(warnings_)

    report_path = report_mod.write_report(
        os.path.join(args.outdir, "report.md"),
        verify_problems=problems,
        video_info_rows=video_rows,
        paired_rows=paired_rows,
        speedup_rows=speedup_rows,
        figures=figures,
        extra_notes=notes,
    )
    print("完成。报告: %s" % report_path)
    for w in warnings_:
        print("  %s" % w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
