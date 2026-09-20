"""生成 prompt 集、sweep 配置、以及评测工具用的 manifest 模板。

三条设计约束：
  1. 每个 prompt 必须落在明确的运动类别里 —— static_background 那类专门用来验证
     "背景闪烁"（推导文档 P4：背景低频能量占比是否会因补偿而下降）。
  2. baseline 与所有稀疏方法共享同一 seed 与同一 prompt，否则配对统计失效。
  3. 类别而非单条 prompt 作为统计单元：同类多条 prompt 才能给出置信区间。

用法：
    python make_experiment.py --outdir manifests --video-dir /workspace/Wan2.1/works
产出：
    prompts.json            16 条 prompt（4 类 × 4 条），含你已在用的 3 条
    sweep_config.json       sweep.py 的输入（模型参数 + run 网格）
    manifest_template.json  video_quality_eval/run_eval.py 的输入骨架
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

# 前 3 条是你已经在跑的 prompt，保留以便与既有 baseline 对齐
PROMPT_CLASSES: Dict[str, List[Dict[str, str]]] = {
    "high_motion": [
        {"id": "p01_cyberpunk_fpv",
         "prompt": "A high-speed cinematic FPV drone dive down a narrow neon-lit cyberpunk alleyway, "
                   "torrential rain streaks, ground puddles dynamically reflecting shifting holographic "
                   "advertisements, high temporal consistency, motion blur."},
        {"id": "p02_rally_car",
         "prompt": "A rally car drifting through a dusty mountain switchback at golden hour, gravel spraying, "
                   "trackside banners whipping in the wind, fast camera pan following the car, sharp motion blur."},
        {"id": "p03_ocean_wave",
         "prompt": "A towering ocean wave breaking over a rocky pier, white foam exploding upward, seabirds "
                   "scattering, handheld camera shaking with the impact, dramatic overcast light."},
        {"id": "p04_dancer_spin",
         "prompt": "A contemporary dancer performing a rapid series of spins on a dark stage, flowing fabric "
                   "trailing behind, moving spotlight, crisp silhouette edges, cinematic slow-motion feel."},
    ],
    "low_motion": [
        {"id": "p05_water_droplet",
         "prompt": "An extreme macro slow-motion shot of a single crystal-clear water droplet piercing a "
                   "mirror-flat liquid surface, precise concentric ripples expanding outward, hyper-detailed "
                   "caustic lighting, crisp edges."},
        {"id": "p06_candle_flame",
         "prompt": "A close-up of a single candle flame in a dark room, gentle flicker, warm falloff on the "
                   "wax surface, still air, extremely subtle motion, shallow depth of field."},
        {"id": "p07_steam_cup",
         "prompt": "A still shot of a ceramic cup of hot tea on a wooden table, thin steam rising and slowly "
                   "curling, soft window light, everything else in the frame motionless."},
        {"id": "p08_snowfall_still",
         "prompt": "A static shot of a snow-covered pine branch, individual snowflakes drifting down slowly, "
                   "overcast winter light, camera locked off, otherwise completely still scene."},
    ],
    "complex_texture": [
        {"id": "p09_forest_bioluminescent",
         "prompt": "A deep-focus shot of a dense mystical nighttime forest, intricate gnarled roots, glowing "
                   "cyan and magenta bioluminescent flora, shimmering floating spores, complex multi-layered "
                   "lighting and shadows."},
        {"id": "p10_autumn_foliage",
         "prompt": "A slow push through dense autumn foliage, thousands of overlapping leaves in red and gold, "
                   "dappled sunlight, intricate branch structure receding into depth, high frequency detail."},
        {"id": "p11_library_shelves",
         "prompt": "A slow tracking shot down an old library aisle, rows of leather-bound books with gilded "
                   "spines, dust motes in a shaft of light, extremely dense repetitive texture."},
        {"id": "p12_coral_reef",
         "prompt": "A macro glide over a vibrant coral reef, branching coral structures, small fish darting "
                   "between polyps, caustic light patterns on every surface, high detail density."},
    ],
    "static_background": [
        # 这一类的用途：背景几乎不动，任何背景亮度/色块的时序脉动都会非常显眼，
        # 是检验"补偿抑制闪烁"信噪比最高的一组。
        {"id": "p13_coffee_table",
         "prompt": "A locked-off shot of a plain white coffee cup on a wooden table against a textured concrete "
                   "wall, a hand places a spoon and withdraws, background stays perfectly static and evenly lit."},
        {"id": "p14_hallway_person",
         "prompt": "A static wide shot of an empty minimalist hallway with flat grey walls and a single door, "
                   "one person walks slowly across the frame from left to right, camera completely still."},
        {"id": "p15_desk_objects",
         "prompt": "A fixed overhead shot of a tidy desk with a notebook, pen and small plant on a plain "
                   "cardboard-colored surface, only the pen rolls slightly, background lighting uniform."},
        {"id": "p16_wall_leaves",
         "prompt": "A still shot facing a plain brick wall with a single potted plant at the base, gentle breeze "
                   "moves only the plant leaves, wall, ground and shadows remain unchanged."},
    ],
}


def build_sweep_config(
    prompts: List[Dict[str, str]],
    seeds: List[int],
    tags: List[Dict[str, Any]],
    video_dir: str,
    model: Dict[str, Any],
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for p in prompts:
        for seed in seeds:
            for t in tags:
                runs.append(
                    {
                        "prompt_id": p["id"],
                        "prompt": p["prompt"],
                        "group": p["group"],
                        "seed": int(seed),
                        "tag": t["tag"],
                        "strategy": t["strategy"],
                        "sparsity": float(t["sparsity"]),
                        "video": os.path.join(video_dir, "%s_%s_s%d.mp4" % (p["id"], t["tag"], seed)),
                    }
                )
    return {"model": model, "runs": runs}


def build_manifest_template(cfg: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment": "wan2.1-1.3B-pisa-sparsity-sweep",
        "baseline_tag": "dense",
        "expected": expected,
        "runs": [{k: v for k, v in r.items() if k != "prompt"} for r in cfg["runs"]],
        "note": "metrics/latency 字段由 sweep.py 回填；直接把它喂给 video_quality_eval/run_eval.py",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="manifests")
    ap.add_argument("--video-dir", default="/workspace/Wan2.1/works")
    ap.add_argument("--seeds", default="12345,67890")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--tags", default="dense:0.0:exact,sparse_drop_08:0.8:hard_drop,pisa_1_08:0.8:order1_compensated")
    args = ap.parse_args(argv)

    prompts: List[Dict[str, str]] = []
    for group, items in PROMPT_CLASSES.items():
        for it in items:
            prompts.append(dict(it, group=group))

    tags = []
    for spec in args.tags.split(","):
        tag, sparsity, strategy = spec.split(":")
        tags.append({"tag": tag, "sparsity": float(sparsity), "strategy": strategy})

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    w, h = args.size.split("*")
    model = {
        "task": "t2v-1.3B",
        "ckpt_dir": "./Wan2.1-T2V-1.3B",
        "size": args.size,
        "frame_num": args.frames,
        "sample_steps": args.steps,
        "sample_shift": 8,
        "sample_guide_scale": 6,
        "offload_model": False,
        "t5_cpu": True,
        "use_prompt_extend": False,
    }
    expected = {"frames": args.frames, "width": int(w), "height": int(h), "fps": 16}

    os.makedirs(args.outdir, exist_ok=True)
    cfg = build_sweep_config(prompts, seeds, tags, args.video_dir, model)
    out = {
        "prompts.json": prompts,
        "sweep_config.json": cfg,
        "manifest_template.json": build_manifest_template(cfg, expected),
    }
    for name, payload in out.items():
        with open(os.path.join(args.outdir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print("prompt 数: %d（4 类 × 4 条）" % len(prompts))
    print("run 数: %d = %d prompt × %d seed × %d 方法配置" % (len(cfg["runs"]), len(prompts), len(seeds), len(tags)))
    est = len(cfg["runs"]) * 4.6 / 60.0
    print("单轮 480p 预估 GPU 小时: 约 %.1f h（按你实测 5.2 s/it × 50 步 ≈ 4.6 分钟/条，不含 720p）"
          % est)
    for name in out:
        print("写出:", os.path.join(args.outdir, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
