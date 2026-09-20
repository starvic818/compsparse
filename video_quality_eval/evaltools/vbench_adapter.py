"""官方 VBench 适配层：background consistency / subject consistency / motion smoothness
（以及可选的 temporal flickering、imaging quality、dynamic degree）。

本模块不重新实现这些指标，只做三件事：
  1. 把我们的 manifest 布局转换成 VBench 需要的目录布局与文件名；
  2. 调用官方 VBench（优先 Python API，失败则退化为 CLI）；
  3. 解析结果 JSON，抽成 (dimension, prompt_index, score) 的平铺结构供配对统计使用。

安装：
    git clone https://github.com/Vchitect/VBench.git
    pip install -e VBench
    # vbench_full_info.json 在 VBench 仓库根目录，用 --vbench-full-info 指定

⚠️ 不同 VBench 版本的 API/CLI 参数与结果文件名会有差异。首次运行时请先执行
    python -c "import vbench, inspect; print(inspect.signature(vbench.VBench.evaluate))"
和  python -m vbench.cli.evaluate -h
再决定用 API 还是用 --vbench-cmd 模板。本模块的默认模板只是最常见的一种写法。

关于 temporal flickering：VBench 自带的 temporal_flickering 维度是全画面的帧间平均绝对差
（MAD），对"背景闪烁 vs 前景运动"是不区分的。我们的 flicker 模块是它的"背景掩码版本"，
两者配合才能把闪烁归因到背景（见 README 的指标对照表）。
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
]


def build_vbench_layout(
    runs: Sequence[Dict[str, Any]],
    outdir: str,
    prompt_index: Dict[str, int],
    copy_files: bool = False,
) -> Dict[str, Any]:
    """按 VBench local 模式的要求组织视频：`{prompt_index}.mp4`（同 prompt 多视频用 -k 后缀）。

    返回 {"dir": outdir, "count": n, "index_map": prompt_index}。
    """
    os.makedirs(outdir, exist_ok=True)
    counters: Dict[int, int] = {}
    n = 0
    for run in runs:
        pid = run.get("prompt_id")
        if pid not in prompt_index:
            raise KeyError("prompt_index 缺少 prompt_id=%s" % pid)
        idx = prompt_index[pid]
        k = counters.get(idx, 0)
        counters[idx] = k + 1
        name = "%d.mp4" % idx if k == 0 else "%d-%d.mp4" % (idx, k)
        src = run["video"]
        dst = os.path.join(outdir, name)
        if copy_files:
            shutil.copyfile(src, dst)
        else:
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        n += 1
    return {"dir": outdir, "count": n, "index_map": dict(prompt_index)}


def run_vbench(
    videos_dir: str,
    output_path: str,
    name: str,
    full_info_dir: str,
    dimension_list: Optional[Sequence[str]] = None,
    device: str = "cuda",
    local: bool = True,
    use_cli: bool = False,
    cli_cmd_template: Optional[str] = None,
    timeout_s: int = 3600 * 6,
) -> Dict[str, Any]:
    """调用官方 VBench。返回 {"mode": ..., "results_json": path, "stdout_tail": ...}。"""
    dims = list(dimension_list or DEFAULT_DIMENSIONS)
    os.makedirs(output_path, exist_ok=True)

    if not use_cli:
        try:
            from vbench import VBench  # type: ignore

            evaluator = VBench(device, full_info_dir, output_path)
            evaluator.evaluate(
                videos_path=videos_dir, name=name, dimension_list=dims, local=local
            )
            return {"mode": "python_api", "stdout_tail": "", "results_json": _find_results_json(output_path, name)}
        except ImportError:
            pass  # 没装 VBench，落回 CLI（用户可能用别的环境跑 CLI）
        except Exception as exc:  # noqa: BLE001
            return {"mode": "python_api", "error": str(exc), "stdout_tail": ""}

    template = cli_cmd_template or (
        "python -m vbench.cli.evaluate --videos_path {videos_path} --dimension {dimensions} "
        "--output_path {output_path} --device {device} --full_json_dir {full_info_dir} --name {name}"
    )
    cmd = (
        template.replace("{videos_path}", videos_dir)
        .replace("{output_path}", output_path)
        .replace("{device}", device)
        .replace("{full_info_dir}", full_info_dir)
        .replace("{name}", name)
        .replace("{dimensions}", ",".join(dims))
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_s)
    return {
        "mode": "cli",
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-40:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-40:]),
        "results_json": _find_results_json(output_path, name),
    }


def _find_results_json(output_path: str, name: str) -> Optional[str]:
    for cand in (
        os.path.join(output_path, "%s_eval_results.json" % name),
        os.path.join(output_path, "%s_results.json" % name),
    ):
        if os.path.isfile(cand):
            return cand
    hits = sorted(glob.glob(os.path.join(output_path, "*.json")), key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


_DIM_ALIASES = {
    "subject consistency": "subject_consistency",
    "background consistency": "background_consistency",
    "motion smoothness": "motion_smoothness",
    "temporal flickering": "temporal_flickering",
    "dynamic degree": "dynamic_degree",
    "imaging quality": "imaging_quality",
    "aesthetic quality": "aesthetic_quality",
    "temporal consistency": "temporal_consistency",
}


def parse_vbench_results(json_path: str) -> List[Dict[str, Any]]:
    """把 VBench 结果 JSON 解析成 [{dimension, prompt_index, score}]。

    VBench 结果结构在不同版本间略有差异，这里做宽松解析：
    只要能在字典里找到 dimension 名 -> 含 'video_results' 或逐个视频分数 的列表即可。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    for key, val in data.items():
        dim = _DIM_ALIASES.get(str(key).strip().lower(), str(key).strip())
        if not isinstance(val, dict):
            continue
        if "video_results" in val and isinstance(val["video_results"], list):
            for i, score in enumerate(val["video_results"]):
                rows.append(
                    {
                        "dimension": dim,
                        "prompt_index": i,
                        "score": _as_float(score),
                        "dimension_mean": _as_float(val.get("num")),
                    }
                )
        elif "video_results" not in val:
            continue
    if not rows:
        raise ValueError(
            "无法从 %s 解析出 VBench 结果；请检查版本差异并手动确认 JSON 结构。" % json_path
        )
    return rows


def _as_float(x: Any) -> float:
    try:
        if isinstance(x, (list, tuple)):
            x = x[0] if x else float("nan")
        return float(x)
    except Exception:
        return float("nan")


def dimension_means(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    import numpy as np

    buckets: Dict[str, List[float]] = {}
    for r in rows:
        s = r.get("score")
        if s is None:
            continue
        if isinstance(s, float) and s != s:
            continue
        buckets.setdefault(r["dimension"], []).append(float(s))
    return {k: float(np.mean(v)) for k, v in buckets.items()}
