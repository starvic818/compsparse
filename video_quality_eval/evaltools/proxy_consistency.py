"""轻量"代理"一致性指标（非官方 VBench，用于日常快速迭代）。

官方 VBench 的 subject consistency 用 DINOv2 特征、background consistency 用 CLIP 特征，
都要跑完整套依赖与分布式流程，代价高。调试期只需要一个能反映"帧间语义漂移"的信号，
所以这里用同样的"帧特征 + 余弦相似度"思路做轻量版本，**数值不等于 VBench**，
只用于趋势判断；最终报告里的数字请用 vbench_adapter 调官方实现。

依赖：torch；'clip' 后端需要 open_clip_torch，'dinov2' 后端需要 timm 或 torch.hub 联网。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .videoutil import decode_video, resize_frames, to_gray_float


def _embed_clip(frames: np.ndarray, device: str, batch: int = 16) -> np.ndarray:
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    from PIL import Image

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, frames.shape[0], batch):
            imgs = [Image.fromarray(f) for f in frames[i : i + batch]]
            x = torch.stack([preprocess(im) for im in imgs]).to(device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def _embed_dinov2(frames: np.ndarray, device: str, batch: int = 8) -> np.ndarray:
    import torch

    try:
        import timm

        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
    except Exception:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device).eval()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, frames.shape[0], batch):
            chunk = frames[i : i + batch].astype(np.float32) / 255.0
            chunk = (chunk - mean) / std
            x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            f = model(x)
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def embed_frames(
    frames: np.ndarray,
    backend: str = "dinov2",
    device: str = "cuda",
    max_side: Optional[int] = 336,
) -> np.ndarray:
    if max_side is not None and max(frames.shape[1:3]) > max_side:
        frames = resize_frames(frames, max_side)
    if backend == "clip":
        return _embed_clip(frames, device)
    if backend == "dinov2":
        return _embed_dinov2(frames, device)
    raise ValueError("未知 backend: %s（可选 clip / dinov2）" % backend)


def mean_pairwise_cosine(feats: np.ndarray, mode: str = "all_pairs") -> float:
    """帧特征的余弦相似度（特征需已 L2 归一化）。越大 = 越一致。"""
    if feats.shape[0] < 2:
        return float("nan")
    sim = feats @ feats.T
    n = sim.shape[0]
    if mode == "adjacent":
        return float(np.mean([sim[i, i + 1] for i in range(n - 1)]))
    if mode == "first_to_rest":
        return float(np.mean(sim[0, 1:]))
    iu = np.triu_indices(n, k=1)
    return float(np.mean(sim[iu]))


def background_crop(frames: np.ndarray, mask: np.ndarray, pad: int = 4) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return frames
    h, w = frames.shape[1], frames.shape[2]
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    return frames[:, y0:y1, x0:x1, :]


def proxy_consistency(
    video_path: str,
    backend: str = "dinov2",
    device: str = "cuda",
    background_only: bool = False,
    ref_video_path: Optional[str] = None,
    mode: str = "all_pairs",
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """单视频的代理一致性分数。background_only=True 时只在背景裁剪区域上算。"""
    frames, meta = decode_video(video_path, max_frames=max_frames)
    if background_only:
        from .flicker import background_mask

        ref = frames
        if ref_video_path:
            ref, _ = decode_video(ref_video_path, max_frames=max_frames)
            ref = resize_frames(ref, max(frames.shape[1], frames.shape[2]))
        mask = background_mask(ref, max_side=None)
        frames = background_crop(frames, mask)

    feats = embed_frames(frames, backend=backend, device=device)
    return {
        "%s_consistency_proxy" % backend: mean_pairwise_cosine(feats, mode=mode),
        "mode": mode,
        "background_only": bool(background_only),
        "n_frames": int(feats.shape[0]),
        "feature_dim": int(feats.shape[1]),
        "note": "代理指标（非官方 VBench 数值）",
    }
