"""环境基线采集：把"这台机器是什么"变成报告附录里可复现的一节。

在云平台上跑一次，产出 env_report.md + env_report.json：
  GPU 型号/显存/驱动/CUDA、当前时钟与降频状态、torch/CUDA/FlashAttention 版本、
  CPU/内存/磁盘余量、关键包版本、以及本次基线的计时口径说明。

为什么值得单独做：加速比结论必须绑定一个明确的软硬件基线。
论文/报告里"我们实现 2.1x 加速"如果不写清 attention 后端（FA2 / FA3 / SDPA-efficient / math）
与是否开启 CPU offload，评审无法判断这个数字是否可复现。

用法: python env_report.py --outdir results/env
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List


def _run(cmd: List[str], timeout: int = 20) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return out.strip()
    except Exception as exc:  # noqa: BLE001
        return "ERROR: %s" % exc


def gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if shutil.which("nvidia-smi") is None:
        if os.name == "nt" and os.path.isfile(r"C:\Windows\System32\nvidia-smi.exe"):
            smi = r"C:\Windows\System32\nvidia-smi.exe"
        else:
            info["available"] = False
            return info
    else:
        smi = "nvidia-smi"

    info["available"] = True
    info["raw_query"] = _run(
        [smi, "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit",
         "--format=csv"]
    )
    info["cuda_version"] = _run([smi, "--query-gpu=driver_version", "--format=csv,noheader"]).strip()
    info["throttle"] = _run([smi, "-q", "-d", "PERFORMANCE"])
    info["topo"] = _run([smi, "topo", "-m"])
    return info


def torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["device_name"] = p.name
            info["capability"] = "%d.%d" % (p.major, p.minor)
            info["total_memory_gb"] = round(p.total_memory / 1024 ** 3, 2)
            info["bf16_supported"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        info["sdpa_available"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: F401

            info["sdpa_backends_api"] = str([b.name for b in SDPBackend])
        except Exception:
            info["sdpa_backends_api"] = "unavailable"
        try:
            from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

            info["flex_attention"] = True
        except Exception:
            info["flex_attention"] = False
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def attention_backend_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    for mod in ("flash_attn", "flash_attn_interface", "xformers", "triton"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = "MISSING"
    # FA3 只在 Hopper (sm90) 上可用；Ada (sm89) 需要 FA2 或 SDPA
    info["note"] = "FA3(flash_attn_interface) 仅支持 Hopper(sm90)；Ada(sm89, RTX40 系) 需用 FA2 或 SDPA/Triton"
    return info


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # 可选

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024 ** 3, 1)
        info["ram_available_gb"] = round(vm.available / 1024 ** 3, 1)
    except Exception:
        info["ram_total_gb"] = "psutil 未安装"
    info["disk"] = _run(["df", "-h"]) or "df 不可用（Windows）"
    return info


def package_info() -> Dict[str, str]:
    keys = ("numpy", "opencv-python", "opencv-python-headless", "pandas", "scipy",
            "matplotlib", "imageio", "imageio-ffmpeg", "decord", "lpips", "timm",
            "open_clip_torch", "diffusers", "transformers", "einops", "flash-attn",
            "vbench", "common-metrics-on-video-quality")
    # 用 importlib.metadata 直接查，避免 `pip list` 在云环境上耗时数十秒
    try:
        from importlib.metadata import PackageNotFoundError, version as pkg_version
    except Exception:  # Python < 3.8
        from importlib_metadata import PackageNotFoundError, version as pkg_version  # type: ignore

    picked: Dict[str, str] = {}
    for k in keys:
        try:
            picked[k] = pkg_version(k)
        except PackageNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            picked[k] = "ERROR: %s" % exc
    picked["_found"] = "%d/%d" % (len(picked), len(keys))
    return picked


def to_markdown(d: Dict[str, Any], title: str) -> str:
    lines = ["## %s\n" % title]
    for k, v in d.items():
        s = str(v)
        if "\n" in s:
            lines.append("### %s\n\n```text\n%s\n```\n" % (k, s))
        else:
            lines.append("- **%s**: %s" % (k, s))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/env")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    data = {
        "gpu": gpu_info(),
        "torch": torch_info(),
        "attention_backends": attention_backend_info(),
        "system": system_info(),
        "packages": package_info(),
    }
    with open(os.path.join(args.outdir, "env_report.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    md = ["# 环境基线（请附在实践报告的实验设置一节）\n",
          "计时口径：以下所有时延均排除模型加载与文本编码，使用 CUDA events + warmup + ≥3 次重复取中位数。\n"]
    md.append(to_markdown(data["gpu"], "GPU"))
    md.append(to_markdown(data["torch"], "PyTorch / CUDA"))
    md.append(to_markdown(data["attention_backends"], "注意力后端"))
    md.append(to_markdown(data["system"], "系统"))
    md.append(to_markdown(data["packages"], "关键依赖版本"))
    with open(os.path.join(args.outdir, "env_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("写入:", os.path.join(args.outdir, "env_report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
