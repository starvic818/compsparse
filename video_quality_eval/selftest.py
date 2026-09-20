"""自检：用合成视频验证指标方向与统计流程是否正确。

做法：造两组合成视频——同一静态背景 + 同一前景运动，其中一组额外叠加"逐帧整体亮度抖动"
（这就是人造的背景闪烁）。正确的指标应当把抖动组判为更差。

运行：python selftest.py            # 会在 ./_selftest 下生成合成数据与完整报告
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evaltools import flicker as flicker_mod  # noqa: E402


def _write_video(path: str, frames: np.ndarray, fps: int = 8) -> str:
    import cv2

    t, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("无法创建 VideoWriter: %s" % path)
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def _synth(seed: int, n_frames: int = 24, size: int = 96, flicker: float = 0.0,
           noise: float = 0.0, blur_sigma: float = 0.0) -> np.ndarray:
    """静态随机纹理背景 + 移动方块。flicker>0 时叠加逐帧整体亮度抖动。"""
    import cv2

    rng = np.random.RandomState(seed)
    base = rng.randint(60, 200, size=(size, size, 3)).astype(np.float32)
    if blur_sigma > 0:
        base = cv2.GaussianBlur(base, (0, 0), blur_sigma)

    frames = np.zeros((n_frames, size, size, 3), dtype=np.uint8)
    box = size // 6
    for t in range(n_frames):
        f = base.copy()
        x = 4 + int((size - box - 8) * t / max(1, n_frames - 1))
        y = size // 2 - box // 2
        f[y : y + box, x : x + box, :] = 240.0
        if flicker > 0:
            gain = 1.0 + flicker * np.sin(2 * np.pi * t / 3.0)
            f = f * gain
        if noise > 0:
            f = f + rng.normal(0, noise, size=f.shape)
        frames[t] = np.clip(f, 0, 255).astype(np.uint8)
    return frames


def main() -> int:
    outdir = os.path.join(HERE, "_selftest")
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    runs = []
    for pi in range(3):
        for seed in (11, 22):
            base_frames = _synth(seed + pi, flicker=0.0)
            jit_frames = _synth(seed + pi, flicker=0.12, noise=1.0)
            dst_dir = os.path.join(outdir, "videos")
            os.makedirs(dst_dir, exist_ok=True)
            base_path = _write_video(os.path.join(dst_dir, "p%d_s%d_dense.avi" % (pi, seed)), base_frames)
            jit_path = _write_video(os.path.join(dst_dir, "p%d_s%d_jitter.avi" % (pi, seed)), jit_frames)
            common = {"prompt_id": "p%d" % pi, "group": "static_bg", "seed": seed}
            runs.append(dict(common, tag="dense", strategy="dense", sparsity=0.0, video=base_path,
                             latency_s=10.0, peak_mem_gb=8.0))
            runs.append(dict(common, tag="jitter", strategy="hard_drop", sparsity=0.8, video=jit_path,
                             latency_s=6.0, peak_mem_gb=6.0))

    manifest = {
        "experiment": "selftest",
        "baseline_tag": "dense",
        "expected": {"frames": 24, "width": 96, "height": 96},
        "runs": runs,
    }
    import json

    mpath = os.path.join(outdir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    import run_eval

    rc = run_eval.main(["--manifest", mpath, "--outdir", os.path.join(outdir, "results"),
                        "--metrics", "verify,flicker", "--max-side", "0", "--n-boot", "2000"])
    if rc != 0:
        print("FAIL: run_eval 返回非 0")
        return 1

    import pandas as pd

    paired_path = os.path.join(outdir, "results", "paired_summary.csv")
    if not os.path.isfile(paired_path) or os.path.getsize(paired_path) == 0:
        print("FAIL: 未生成 paired_summary.csv（配对统计失效）")
        return 1
    paired = pd.read_csv(paired_path)
    fi = paired[paired.metric == "flicker_index"]
    bg = paired[paired.metric == "bg_l1_mean"]
    dc = paired[paired.metric == "patch_dc_var_mean"]
    print("\n--- 自检结果 ---")
    print(fi[["metric", "tag", "base_mean", "method_mean", "delta_mean", "ci_low", "ci_high", "verdict"]].to_string(index=False))

    ok = True
    if fi.empty or not (fi.method_mean.iloc[0] > fi.base_mean.iloc[0] * 1.05):
        print("FAIL: 抖动组的 flicker_index 未明显高于 baseline")
        ok = False
    if fi.empty or fi.verdict.iloc[0] != "degraded":
        print("FAIL: 抖动组未被判为 degraded")
        ok = False
    if dc.empty or dc.delta_mean.iloc[0] <= 0:
        print("FAIL: patch_dc_var 未检测到亮度脉动")
        ok = False
    if bg.empty or bg.delta_mean.iloc[0] <= 0:
        print("FAIL: bg_l1 未检测到帧间残差上升")
        ok = False

    # 掩码合理性：移动方块约占面积 1/36，掩码不应吃掉半个画面
    prof = flicker_mod.flicker_from_files(runs[0]["video"], max_side=None)
    if not (0.05 < prof["bg_mask_ratio"] < 0.6):
        print("FAIL: 背景掩码比例异常 %.3f" % prof["bg_mask_ratio"])
        ok = False
    else:
        print("背景掩码占比 %.3f（合理）" % prof["bg_mask_ratio"])

    print("自检结论:", "PASS" if ok else "FAIL")
    print("产物目录:", os.path.join(outdir, "results"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
