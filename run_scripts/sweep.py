"""批量生成 + 计时 + 回填 manifest（一次加载，断点续跑）。

为什么需要它：你现在的流程是每条命令重新加载一次模型（约 2 分钟），
一轮 96 条配置就会白烧 3 小时。这个脚本让模型只加载一次。

三种模式：
    --dry-run                只打印计划、写 manifest.json（不需要 GPU，也不需要装 wan）
    --runner inproc          进程内加载 WanT2V 一次，循环生成（推荐，计时最准）
    --runner subprocess      每条配置起一个 python generate.py（与你现在的流程一致，
                             但计时会包含模型加载，只能用于粗略对照）

计时口径（写进报告的实验设置一节）：
    latency_s   = 采样循环的耗时（CUDA events / 墙钟），不含模型加载与文本编码
    peak_mem_gb = torch.cuda.max_memory_allocated() 的峰值（仅 inproc 模式）

用法：
    python make_experiment.py --outdir manifests --video-dir /workspace/Wan2.1/works
    python sweep.py --config manifests/sweep_config.json --outdir results_sweep --dry-run
    python sweep.py --config manifests/sweep_config.json --outdir results_sweep --runner inproc --limit 2

稀疏方法接入：在 sweep_config.json 的每个 tag 里写
    "patch": "my_sparse_patch:install",  "patch_kwargs": {"sparsity": 0.8, "strategy": "order1"}
脚本会在构建 pipeline 之前 import 该函数并调用它（返回 None），由你的代码去替换注意力实现。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ 视频保存

def save_video_robust(video, path: str, fps: int = 16) -> str:
    """显式做 uint8 转换的保存函数，绕开 Wan 原版 cache_video 的 dtype 问题
    （日志里的 `result type Float can't be cast to the desired output type Byte`）。

    video 支持 [C,T,H,W] 或 [T,H,W,C]，取值 [-1,1] 或 [0,1]。
    """
    import numpy as np
    import torch

    if isinstance(video, torch.Tensor):
        arr = video.detach().float().cpu().numpy()
    else:
        arr = np.asarray(video, dtype="float32")
    # 统一成 [T,H,W,C]
    if arr.ndim != 4:
        raise ValueError("期望 4 维张量，得到 %s" % (arr.shape,))
    if arr.shape[0] in (1, 3, 4) and arr.shape[1] > arr.shape[0]:
        arr = np.transpose(arr, (1, 2, 3, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.min() < -0.05:
        arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    frames = (arr * 255.0 + 0.5).astype("uint8")      # 显式 round + 转换，避免就地写回 uint8

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import cv2

        t, h, w, _ = frames.shape
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("cv2 VideoWriter 打开失败")
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
    except Exception:
        import imageio

        imageio.mimsave(path, list(frames), fps=fps)
    return path


def probe_frames(path: str) -> Optional[int]:
    if not os.path.isfile(path):
        return None
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok = cap.isOpened()
        cap.release()
        return n if ok else None
    except Exception:
        return None


# ------------------------------------------------------------------ 计划

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cli_args_for(run: Dict[str, Any], model: Dict[str, Any]) -> List[str]:
    """subprocess 模式拼出的 generate.py 命令（也可用于打印给人的检查）。"""
    args = [
        "python", "generate.py",
        "--task", str(model.get("task", "t2v-1.3B")),
        "--size", str(model.get("size", "832*480")),
        "--frame_num", str(model.get("frame_num", 81)),
        "--ckpt_dir", str(model.get("ckpt_dir", "./Wan2.1-T2V-1.3B")),
        "--sample_steps", str(model.get("sample_steps", 50)),
        "--sample_shift", str(model.get("sample_shift", 8)),
        "--sample_guide_scale", str(model.get("sample_guide_scale", 6)),
        "--base_seed", str(run["seed"]),
        "--offload_model", str(bool(model.get("offload_model", False))),
        "--t5_cpu", str(bool(model.get("t5_cpu", True))),
        "--use_prompt_extend", str(bool(model.get("use_prompt_extend", False))),
        "--prompt", str(run["prompt"]),
        "--save_file", str(run["video"]),
    ]
    return args


def plan_table(cfg: Dict[str, Any], limit: Optional[int]) -> List[Dict[str, Any]]:
    runs = cfg["runs"]
    return runs[:limit] if limit else runs


def write_manifest(cfg: Dict[str, Any], rows: List[Dict[str, Any]], path: str, started_ts: float) -> str:
    manifest = {
        "experiment": "wan2.1 sweep",
        "started_ts": started_ts,
        "baseline_tag": "dense",
        "expected": {
            "frames": int(cfg["model"].get("frame_num", 81)),
            "width": int(str(cfg["model"].get("size", "832*480")).split("*")[0]),
            "height": int(str(cfg["model"].get("size", "832*480")).split("*")[1]),
            "fps": 16,
        },
        "runs": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def append_timing(path: str, row: Dict[str, Any]) -> None:
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------------ 执行

def build_pipeline(model: Dict[str, Any]):
    from wan import WanT2V

    class _Args:
        pass

    a = _Args()
    for k, v in model.items():
        setattr(a, k, v)
    a.device = "cuda" if _cuda_ok() else "cpu"
    a.ulysses_size = 1
    a.ring_size = 1
    a.t5_fsdp = False
    a.dit_fsdp = False
    return WanT2V(a)


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def apply_patch(spec: str, kwargs: Dict[str, Any]) -> None:
    mod_name, _, fn_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name or "install")
    fn(**(kwargs or {}))
    print("   已应用 patch: %s(%s)" % (spec, kwargs or {}))


def run_inproc(cfg: Dict[str, Any], runs: List[Dict[str, Any]], outdir: str, resume: bool) -> List[Dict[str, Any]]:
    import torch

    model = cfg["model"]
    rows: List[Dict[str, Any]] = []
    current_patch = None
    pipe = None

    for i, run in enumerate(runs, 1):
        patch = run.get("patch")
        if patch != current_patch:
            if pipe is not None:
                print("   [info] 方法配置变化，重建 pipeline（会多花一次加载时间）")
            current_patch = patch
            if patch:
                apply_patch(patch, run.get("patch_kwargs", {}))
            pipe = build_pipeline(model)

        frames = probe_frames(run["video"]) if resume else None
        expected_frames = int(model.get("frame_num", 81))
        if resume and frames and abs(frames - expected_frames) <= 1:
            print("   [%d/%d] 跳过（已存在且帧数正确）: %s" % (i, len(runs), run["video"]))
            rows.append(dict(run, latency_s=None, peak_mem_gb=None, skipped=True))
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_ev, end_ev = torch.cuda.Event(True), torch.cuda.Event(True)
            start_ev.record()
        t0 = time.time()
        video = pipe.generate(
            prompt=run["prompt"],
            size=model.get("size", "832*480"),
            frame_num=int(model.get("frame_num", 81)),
            seed=int(run["seed"]),
            t5_cpu=bool(model.get("t5_cpu", True)),
            offload_model=bool(model.get("offload_model", False)),
        )
        if torch.cuda.is_available():
            end_ev.record()
            torch.cuda.synchronize()
            latency = start_ev.elapsed_time(end_ev) / 1000.0
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        else:
            latency = time.time() - t0
            peak = float("nan")

        save_video_robust(video, run["video"], fps=16)
        n = probe_frames(run["video"])
        print("   [%d/%d] %s %s seed=%s -> %.1fs, peak=%.2fGB, frames=%s"
              % (i, len(runs), run["prompt_id"], run["tag"], run["seed"], latency, peak, n))
        rec = dict(run, latency_s=round(latency, 3), peak_mem_gb=round(peak, 3),
                   n_frames=n, steps=model.get("sample_steps"), skipped=False)
        rows.append(rec)
        append_timing(os.path.join(outdir, "timing.csv"), rec)
    return rows


def run_subprocess(cfg: Dict[str, Any], runs: List[Dict[str, Any]], outdir: str, resume: bool,
                   dry_run: bool) -> List[Dict[str, Any]]:
    model = cfg["model"]
    rows: List[Dict[str, Any]] = []
    for i, run in enumerate(runs, 1):
        cmd = cli_args_for(run, model)
        if dry_run:
            print("   [%d/%d] %s" % (i, len(runs), " ".join(cmd[:8]) + " ... --save_file " + run["video"]))
            rows.append(dict(run, latency_s=None, peak_mem_gb=None))
            continue
        frames = probe_frames(run["video"]) if resume else None
        if resume and frames and abs(frames - int(model.get("frame_num", 81))) <= 1:
            print("   [%d/%d] 跳过（已存在）: %s" % (i, len(runs), run["video"]))
            rows.append(dict(run, latency_s=None, peak_mem_gb=None, skipped=True))
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        print("   [%d/%d] rc=%d wall=%.1fs %s" % (i, len(runs), proc.returncode, wall, run["prompt_id"]))
        if proc.returncode != 0:
            print("      stderr tail: %s" % "\n".join((proc.stderr or "").splitlines()[-5:]))
        rec = dict(run, latency_s=round(wall, 3), peak_mem_gb=None,
                   n_frames=probe_frames(run["video"]), steps=model.get("sample_steps"))
        rows.append(rec)
        append_timing(os.path.join(outdir, "timing.csv"), rec)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="results_sweep")
    ap.add_argument("--runner", default="inproc", choices=["inproc", "subprocess"])
    ap.add_argument("--dry-run", action="store_true", help="只打印计划并写 manifest.json（无需 GPU）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（smoke test 用）")
    ap.add_argument("--no-resume", action="store_true", help="不跳过已存在的输出")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    os.makedirs(args.outdir, exist_ok=True)
    runs = plan_table(cfg, args.limit or None)
    started_ts = time.time()

    mode = "dry-run" if args.dry_run else args.runner
    total_gb = 0.0
    print("模式=%s  配置数=%d  模型=%s" % (mode, len(runs), cfg["model"].get("ckpt_dir")))
    print("输出目录=%s  续跑=%s" % (args.outdir, not args.no_resume))

    if args.dry_run:
        rows = run_subprocess(cfg, runs, args.outdir, resume=False, dry_run=True)
    elif args.runner == "inproc":
        rows = run_inproc(cfg, runs, args.outdir, resume=not args.no_resume)
    else:
        rows = run_subprocess(cfg, runs, args.outdir, resume=not args.no_resume, dry_run=False)

    # 回填 manifest（供 video_quality_eval/run_eval.py 使用）
    keep = ("prompt_id", "group", "seed", "tag", "strategy", "sparsity", "video",
            "latency_s", "peak_mem_gb")
    manifest_rows = [{k: r.get(k) for k in keep} for r in rows]
    mpath = write_manifest(cfg, manifest_rows, os.path.join(args.outdir, "manifest.json"), started_ts)
    print("\nmanifest:", mpath)
    print("timing.csv:", os.path.join(args.outdir, "timing.csv"))
    print("下一步：python video_quality_eval/run_eval.py --manifest %s --outdir %s/eval --metrics verify,flicker"
          % (mpath, args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
