"""FVD / KID：作为"最终小规模补充指标"，不是主指标。

为什么只当补充：
  1. FVD 是**分布距离**，需要足够多的样本才稳定（经验上 <64 段视频时方差极大），
     而你的 480p/81f 一次生成就要 4~5 分钟，100 段 = 十几个小时。
  2. FVD 对"局部背景闪烁"不敏感：几十帧的背景轻微脉动会被全局分布距离平均掉。
     所以"PISA 抑制背景闪烁"这个卖点必须用 flicker 模块的配对指标来支撑，
     FVD 只用来证明"整体分布没有崩"。

本模块提供：
  frechet_distance / polynomial_mmd      —— 纯数学部分（可独立验证）
  make_extractor_*                       —— 特征提取器工厂（默认 torchvision R3D）
  fvd_between_sets                       —— 两个视频集合之间的 FVD（相对 FVD）
  bootstrap_fvd_ci                       —— 按 prompt 配对重采样的置信区间

⚠️ 官方 FVD 使用 I3D(Kinetics-400) 特征。若你安装了 VBench 依赖
   `common_metrics_on_video_quality`，请用 make_extractor_common_metrics() 以拿到可比数值；
   未安装时退回 R3D 特征，此时请在报告中写明"R3D 特征版相对 FVD，非官方 I3D-FVD"。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .videoutil import decode_video

FeatureExtractor = Callable[[Sequence[str]], np.ndarray]


# --------------------------------------------------------------------- 距离

def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    """对称 PSD 矩阵的平方根（特征值截断，避免 sqrtm 出现微小复数）。"""
    vals, vecs = np.linalg.eigh((a + a.T) / 2.0)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet 距离（FVD 的核心）。feats_*: [N, d]。"""
    a = np.asarray(feats_a, dtype=np.float64)
    b = np.asarray(feats_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("特征必须是二维 [N, d]")
    if a.shape[0] < 2 or b.shape[0] < 2:
        return float("nan")
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    ca = np.cov(a, rowvar=False) + np.eye(a.shape[1]) * eps
    cb = np.cov(b, rowvar=False) + np.eye(b.shape[1]) * eps
    diff = mu_a - mu_b
    covmean = _sqrtm_psd(_sqrtm_psd(ca) @ cb @ _sqrtm_psd(ca))
    covmean = _sqrtm_psd(covmean @ covmean)  # 再投影一次，抵消数值误差
    val = float(diff @ diff + np.trace(ca) + np.trace(cb) - 2.0 * np.trace(covmean))
    return max(val, 0.0)


def polynomial_mmd(feats_a: np.ndarray, feats_b: np.ndarray, degree: int = 3, gamma: Optional[float] = None) -> float:
    """KID：无偏多项式核 MMD。样本量小时比 FVD 稳定得多。"""
    a = np.asarray(feats_a, dtype=np.float64)
    b = np.asarray(feats_b, dtype=np.float64)
    d = a.shape[1]
    gamma = gamma or (1.0 / d)

    def k(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (gamma * (x @ y.T) + 1.0) ** degree

    ka, kb = k(a, a), k(b, b)
    kab = k(a, b)
    n, m = a.shape[0], b.shape[0]
    sum_a = (ka.sum() - np.trace(ka)) / (n * (n - 1))
    sum_b = (kb.sum() - np.trace(kb)) / (m * (m - 1))
    return float(sum_a + sum_b - 2.0 * kab.mean())


# --------------------------------------------------------------------- 特征

def make_extractor_r3d(device: str = "cuda", size: int = 112, max_frames: int = 32) -> FeatureExtractor:
    """用 torchvision 预训练 R3D-18 抽特征（相对比较用，非官方 I3D-FVD）。"""

    def extractor(paths: Sequence[str]) -> np.ndarray:
        import torch
        import torchvision

        weights = torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        model = torchvision.models.video.r3d_18(weights=weights).to(device).eval()
        model.fc = torch.nn.Identity()
        feats: List[np.ndarray] = []
        for p in paths:
            frames, _ = decode_video(p, max_frames=max_frames)
            x = _to_clip_tensor(frames, size=size, device=device)
            with torch.no_grad():
                f = model(x).float().cpu().numpy().reshape(-1)
            feats.append(f)
            del x
        return np.stack(feats, axis=0)

    extractor.name = "r3d18_kinetics400"  # type: ignore[attr-defined]
    return extractor


def make_extractor_common_metrics(device: str = "cuda") -> FeatureExtractor:
    """官方 FVD 路径：使用 common_metrics_on_video_quality（VBench 依赖）的 I3D 特征。"""
    try:
        from common_metrics_on_video_quality import calculate_fvd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "未安装 common_metrics_on_video_quality。官方 I3D-FVD 需要它；"
            "否则请用 make_extractor_r3d() 并注明是相对 FVD。"
        ) from exc

    def extractor(paths: Sequence[str]) -> np.ndarray:
        fn = getattr(calculate_fvd, "get_fvd_features", None)
        if fn is None:
            raise RuntimeError(
                "common_metrics_on_video_quality 的 API 与本适配器不匹配（不同版本函数名不同）。"
                "请查看该包内 calculate_fvd 模块实际提供的函数名，或改用 --fvd-extractor r3d。"
            )
        feats = fn(list(paths), device=device)
        return np.asarray(feats, dtype=np.float64)

    extractor.name = "i3d_common_metrics"  # type: ignore[attr-defined]
    return extractor


def _to_clip_tensor(frames: np.ndarray, size: int, device: str):
    import torch

    x = torch.from_numpy(frames.astype(np.float32) / 255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,T,H,W]
    x = torch.nn.functional.interpolate(
        x, size=(x.shape[2], size, size), mode="trilinear", align_corners=False
    )
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=x.device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=x.device).view(1, 3, 1, 1, 1)
    return ((x - mean) / std).to(device)


# --------------------------------------------------------------------- 入口

def fvd_between_sets(
    paths_a: Sequence[str],
    paths_b: Sequence[str],
    extractor: Optional[FeatureExtractor] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """两个视频集合之间的距离。A=dense/baseline，B=稀疏方法 => 即"相对 FVD"。"""
    ext = extractor or make_extractor_r3d(device=device)
    fa = ext(list(paths_a))
    fb = ext(list(paths_b))
    return {
        "extractor": getattr(ext, "name", "custom"),
        "n_a": int(fa.shape[0]),
        "n_b": int(fb.shape[0]),
        "fvd": frechet_distance(fa, fb),
        "kid": polynomial_mmd(fa, fb),
        "note": "两生成集合之间的距离（相对 FVD）；数值随特征提取器与样本量变化，请勿跨设置比较",
    }


def bootstrap_fvd_ci(
    paths_a: Sequence[str],
    paths_b: Sequence[str],
    extractor: Optional[FeatureExtractor] = None,
    device: str = "cuda",
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """按样本（prompt）配对重采样：每轮同时抽取 A、B 的同一批下标。

    注意：对 FVD 做 bootstrap 需要重复计算协方差平方根，代价较高（但特征已缓存，可接受）。
    """
    ext = extractor or make_extractor_r3d(device=device)
    fa = ext(list(paths_a))
    fb = ext(list(paths_b))
    n = min(fa.shape[0], fb.shape[0])
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        vals.append(frechet_distance(fa[idx], fb[idx]))
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        "fvd_point": frechet_distance(fa, fb),
        "fvd_ci_low": float(np.percentile(arr, 100 * alpha / 2)) if arr.size else float("nan"),
        "fvd_ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))) if arr.size else float("nan"),
        "n_boot_ok": int(arr.size),
    }
