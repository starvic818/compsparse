"""闪烁（temporal flicker）指标：面向"背景不应变化"的视频配对对比。

设计原则（与 VBench 互补的地方）：
  1. 指向性：背景闪烁的定义是"静止区域在时间上不应变化"，所以指标必须在 **背景掩码** 上
     计算，而不是全画面平均（全画面平均会被前景运动淹没）。
  2. 配对：掩码不能来自被评测视频自己（否则"闪得越厉害 -> 掩码越乱"是循环论证）。
     本模块用同一 prompt+seed 的 baseline 视频提取低运动区域作为共享掩码，
     两个方法在同一批像素上比较。
  3. 分层：三种互补度量
       bg_l1        帧间像素差（对高频闪烁敏感）
       warp_resid   光流把 t+1 对齐回 t 后的残差（对背景漂移/非刚性变化敏感）
       patch_dc_var 低通后每个 patch 平均亮度的时间方差（对"明暗/色块脉动"，即人眼最
                    典型的闪烁感知最敏感）
  4. 可解释：输出原始值、相对 baseline 的比值、以及合成 flicker_index
     （三个比值的几何平均，1.0 = 与 baseline 同等稳定，>1 = 更闪）。

注意：这是自研闪烁指标，与官方 VBench 的 background consistency 不是同一个量。
后者是 CLIP 特征层面的帧间相似度（越大越好），本模块是像素/低频亮度层面的时间不稳定
（越小越好）。两者应同向但不必一致，报告中建议并列。

依赖：numpy, opencv-python（可选：lpips + torch）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .videoutil import decode_video, resize_frames, to_gray_float, to_rgb_float

DEFAULT_MAX_SIDE = 512
_PRIMARY = ("bg_l1_mean", "warp_resid_mean", "patch_dc_var_mean")


def _odd(v: int) -> int:
    v = int(v)
    return v if v % 2 == 1 else v + 1


def background_mask(
    ref_frames: np.ndarray,
    percentile: float = 30.0,
    blur: int = 5,
    dilate: int = 3,
    min_area_ratio: float = 0.002,
    max_side: Optional[int] = DEFAULT_MAX_SIDE,
    max_ratio: float = 0.6,
) -> np.ndarray:
    """从 baseline 视频提取"低时间活跃度"像素作为背景掩码，返回 bool [H,W]。

    percentile=30 表示取时间活跃度最低的 30% 像素（按**排序位次**取，而不是按阈值取）。
    这一点很重要：真实视频里大量静止像素的活跃度完全相等（都接近 0），
    用 `activity <= percentile(activity)` 会把这一大片并列值全部选进来，
    掩码可能瞬间涨到 60% 以上。改用位次选择后，掩码比例稳定在设定值附近。
    """
    import cv2

    if max_side is not None and max(ref_frames.shape[1:3]) > max_side:
        ref_frames = resize_frames(ref_frames, max_side)

    gray = to_gray_float(ref_frames)
    if gray.shape[0] < 2:
        return np.ones(gray.shape[1:], dtype=bool)

    activity = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    if blur > 1:
        activity = cv2.GaussianBlur(activity, (_odd(blur), _odd(blur)), 0)

    flat = activity.reshape(-1)
    order = np.argsort(flat, kind="stable")

    def _rank_mask(frac: float) -> np.ndarray:
        n_sel = int(round(max(0.0, min(1.0, frac)) * flat.size))
        n_sel = max(1, min(n_sel, flat.size))
        sel = np.zeros(flat.size, dtype=bool)
        sel[order[:n_sel]] = True
        return sel.reshape(activity.shape).astype(np.uint8)

    mask = _rank_mask(percentile / 100.0)

    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_lab > 1:
        keep = np.zeros_like(mask)
        for lab in range(1, n_lab):
            if stats[lab, cv2.CC_STAT_AREA] >= min_area_ratio * mask.size:
                keep[labels == lab] = 1
        mask = keep

    if dilate and dilate > 1:
        mask = cv2.dilate(mask, np.ones((_odd(dilate), _odd(dilate)), np.uint8), iterations=1)

    # 形态学可能让掩码膨胀，用位次上限收回来，保证背景掩码不会吃掉半个画面
    if mask.mean() > max_ratio:
        mask = (mask & _rank_mask(max_ratio)).astype(np.uint8)

    if mask.sum() < 0.01 * mask.size:
        mask = _rank_mask(max(percentile, 50.0) / 100.0)
    return mask.astype(bool)


def flicker_bg_l1(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """背景掩码内的帧间 L1 差的均值 / 标准差 / p95。"""
    if gray.shape[0] < 2:
        return {"bg_l1_mean": 0.0, "bg_l1_std": 0.0, "bg_l1_p95": 0.0}
    diff = np.abs(np.diff(gray, axis=0))
    if mask.any():
        per_t = diff[:, mask].mean(axis=1)
    else:
        per_t = diff.reshape(diff.shape[0], -1).mean(axis=1)
    return {
        "bg_l1_mean": float(per_t.mean()),
        "bg_l1_std": float(per_t.std()),
        "bg_l1_p95": float(np.percentile(per_t, 95)),
    }


def flicker_warp_residual(
    gray: np.ndarray,
    mask: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> Dict[str, float]:
    """光流对齐残差：Farneback 估计 t->t+1 的流，把 t+1 反向 warp 回 t 后取残差。

    背景只是整体平移/缓慢漂移时残差接近 0；发生亮度/纹理闪烁（不服从运动模型）时残差
    无法被解释掉，指标升高。光流有噪声地板，故只在同一实现/分辨率下互相比较。
    """
    import cv2

    t = gray.shape[0]
    if t < 2:
        return {"warp_resid_mean": 0.0, "warp_resid_std": 0.0, "warp_resid_p95": 0.0}

    h, w = gray.shape[1], gray.shape[2]
    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    per_t = np.empty(t - 1, dtype=np.float32)

    for i in range(t - 1):
        a = (gray[i] * 255.0).astype(np.uint8)
        b = (gray[i + 1] * 255.0).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0
        )
        warped = cv2.remap(
            gray[i + 1],
            gx - flow[..., 0],
            gy - flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        resid = np.abs(warped - gray[i])
        per_t[i] = resid[mask].mean() if mask.any() else resid.mean()

    return {
        "warp_resid_mean": float(per_t.mean()),
        "warp_resid_std": float(per_t.std()),
        "warp_resid_p95": float(np.percentile(per_t, 95)),
    }


def flicker_patch_dc(
    frames: np.ndarray,
    mask: np.ndarray,
    grid: int = 4,
    lowpass_sigma: float = 2.0,
    min_coverage: float = 0.15,
    lowfreq_frac: float = 0.2,
) -> Dict[str, float]:
    """背景分块 DC（低频亮度）在时间上的方差。

    先低通（抑制细节与压缩噪声，只留低频亮度），切成 grid×grid 个 patch，每个 patch 只
    统计掩码内像素的平均亮色，得到一条时间序列；对该序列求时间方差，再对背景 patch 取
    均值/中位数/90 分位。最接近人眼对"背景忽明忽暗"的判断。

    额外输出 patch_dc_lowfreq_ratio：把每个 patch 的 DC 序列做功率谱，取最低
    lowfreq_frac 比例频率成分的能量占比并平均。**这是推导文档 P4 的直接检验量**：
    硬丢弃造成的偏差能量集中在低频，一阶补偿后应向高频转移，该比值随之下降。
    """
    import cv2

    rgb = to_rgb_float(frames)
    if lowpass_sigma and lowpass_sigma > 0:
        ksize = _odd(int(round(lowpass_sigma * 4)))
        rgb = np.stack(
            [cv2.GaussianBlur(rgb[i], (ksize, ksize), lowpass_sigma) for i in range(rgb.shape[0])],
            axis=0,
        )

    t, h, w = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    empty = {
        "patch_dc_var_mean": 0.0,
        "patch_dc_var_median": 0.0,
        "patch_dc_var_p90": 0.0,
        "patch_dc_lowfreq_ratio": 0.0,
        "n_bg_patches": 0.0,
    }
    if t < 2:
        return empty

    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    series = []

    for bi in range(grid):
        for bj in range(grid):
            sl = (slice(ys[bi], ys[bi + 1]), slice(xs[bj], xs[bj + 1]))
            m = mask[sl]
            if m.size == 0 or m.mean() < min_coverage:
                continue
            block = rgb[:, sl[0], sl[1], :].reshape(t, m.size, 3)
            series.append(block[:, m.reshape(-1), :].mean(axis=1))

    if not series:
        return empty

    stacked = np.stack(series, axis=0)
    per_patch = stacked.var(axis=1).mean(axis=1)

    # 功率谱：低频能量占比（P4 检验量）
    x = stacked.mean(axis=-1)                       # [P, T] 通道平均后的 DC 序列
    x = x - x.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2     # [P, K]
    if power.shape[1] > 2:
        k_low = max(1, int(round(lowfreq_frac * (power.shape[1] - 1))))
        total = power[:, 1:].sum(axis=1)
        low = power[:, 1 : 1 + k_low].sum(axis=1)
        valid = total > 1e-16
        lowfreq_ratio = float(np.mean(low[valid] / total[valid])) if valid.any() else 0.0
    else:
        lowfreq_ratio = 0.0

    return {
        "patch_dc_var_mean": float(per_patch.mean()),
        "patch_dc_var_median": float(np.median(per_patch)),
        "patch_dc_var_p90": float(np.percentile(per_patch, 90)),
        "patch_dc_lowfreq_ratio": lowfreq_ratio,
        "n_bg_patches": float(len(per_patch)),
    }


def flicker_lpips(
    frames: np.ndarray,
    mask: np.ndarray,
    net: str = "alex",
    device: str = "cuda",
) -> Dict[str, float]:
    """可选：背景裁剪区域的帧间 LPIPS（需要 pip install lpips torch）。"""
    try:
        import lpips as lpips_pkg
        import torch
    except Exception as exc:  # noqa: BLE001
        raise ImportError("需要 lpips 与 torch: pip install lpips") from exc

    rgb = to_rgb_float(frames)
    ys, xs = np.where(mask)
    if ys.size == 0 or rgb.shape[0] < 2:
        return {"lpips_bg_mean": float("nan"), "lpips_bg_std": float("nan")}
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    model = lpips_pkg.LPIPS(net=net).to(device).eval()
    vals = []
    with torch.no_grad():
        for i in range(rgb.shape[0] - 1):
            a = torch.from_numpy(rgb[i, y0:y1, x0:x1, :]).permute(2, 0, 1).unsqueeze(0)
            b = torch.from_numpy(rgb[i + 1, y0:y1, x0:x1, :]).permute(2, 0, 1).unsqueeze(0)
            ta = (a.to(device) * 2 - 1).float()
            tb = (b.to(device) * 2 - 1).float()
            vals.append(float(model(ta, tb).item()))
    arr = np.asarray(vals, dtype=np.float32)
    return {"lpips_bg_mean": float(arr.mean()), "lpips_bg_std": float(arr.std())}


def _safe_ratio(a: float, b: float, eps: float = 1e-8) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < eps:
        return float("nan")
    return float(a / b)


def _align_mask(mask: np.ndarray, shape_hw) -> np.ndarray:
    import cv2

    if mask.shape == tuple(shape_hw):
        return mask
    m8 = (mask.astype(np.uint8) * 255)[:, :, None]
    m8 = cv2.resize(m8, (int(shape_hw[1]), int(shape_hw[0])), interpolation=cv2.INTER_NEAREST)
    return m8[..., 0] > 127


def flicker_profile(
    frames: np.ndarray,
    ref_frames: Optional[np.ndarray] = None,
    max_side: Optional[int] = DEFAULT_MAX_SIDE,
    mask_percentile: float = 30.0,
    patch_grid: int = 4,
    lowpass_sigma: float = 2.0,
    with_lpips: bool = False,
    lpips_device: str = "cuda",
) -> Dict[str, Any]:
    """计算一段视频的闪烁画像（详见模块 docstring）。"""
    if max_side is not None and max(frames.shape[1:3]) > max_side:
        frames = resize_frames(frames, max_side)
    if ref_frames is not None and max_side is not None and max(ref_frames.shape[1:3]) > max_side:
        ref_frames = resize_frames(ref_frames, max_side)

    if ref_frames is not None:
        ref_frames = resize_frames(ref_frames, max(frames.shape[1], frames.shape[2]))
        mask = background_mask(ref_frames, percentile=mask_percentile, max_side=None)
        mask_source = "reference"
    else:
        mask = background_mask(frames, percentile=mask_percentile, max_side=None)
        mask_source = "self"
    mask = _align_mask(mask, frames.shape[1:3])

    gray = to_gray_float(frames)
    out: Dict[str, Any] = {"mask_source": mask_source, "bg_mask_ratio": float(mask.mean())}
    out.update(flicker_bg_l1(gray, mask))
    out.update(flicker_warp_residual(gray, mask))
    out.update(flicker_patch_dc(frames, mask, grid=patch_grid, lowpass_sigma=lowpass_sigma))
    if with_lpips:
        out.update(flicker_lpips(frames, mask, device=lpips_device))

    if ref_frames is not None:
        ref_gray = to_gray_float(ref_frames)
        ref_vals: Dict[str, float] = {}
        ref_vals.update(flicker_bg_l1(ref_gray, mask))
        ref_vals.update(flicker_warp_residual(ref_gray, mask))
        ref_vals.update(
            flicker_patch_dc(ref_frames, mask, grid=patch_grid, lowpass_sigma=lowpass_sigma)
        )
        for k, v in ref_vals.items():
            out["ref_" + k] = float(v)
        for k in _PRIMARY:
            out[k.replace("_mean", "") + "_ratio"] = _safe_ratio(float(out[k]), float(ref_vals[k]))
        ratios = [out[k.replace("_mean", "") + "_ratio"] for k in _PRIMARY]
        ratios = [r for r in ratios if np.isfinite(r) and r > 0]
        out["flicker_index"] = (
            float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")
        )
    return out


def flicker_from_files(
    video_path: str,
    ref_video_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    frames, meta = decode_video(video_path, max_frames=max_frames)
    ref_frames = None
    if ref_video_path:
        ref_frames, _ = decode_video(ref_video_path, max_frames=max_frames)
    out = flicker_profile(frames, ref_frames=ref_frames, **kwargs)
    out["n_frames"] = meta["n_frames"]
    return out
