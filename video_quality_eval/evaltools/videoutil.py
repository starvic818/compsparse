"""视频 I/O、探测与完整性校验。

为什么需要这一层：Wan2.1 的生成脚本先打印 "Saving generated video to ..." 再真正写文件，
写盘失败时只留一行 `cache_video failed` 日志；它还有 cache-file 机制，同名文件已存在会
直接复用旧文件。于是会出现"文件存在但内容不是本次结果"。所以算指标之前必须校验：
存在性、可解码性、帧数、分辨率、fps、修改时间，以及跨 run 的内容 hash 重复。

依赖：numpy；解码后端优先 opencv，其次 imageio。
"""

from __future__ import annotations

import datetime
import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk_size)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _read_cv2(path: str, max_frames: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("opencv 无法打开视频: %s" % path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise IOError("opencv 解码得到 0 帧: %s" % path)
    return np.stack(frames, axis=0), {"fps": fps, "backend": "cv2"}


def _read_imageio(path: str, max_frames: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    import imageio

    reader = imageio.get_reader(path)
    meta: Dict[str, Any] = {}
    try:
        meta = dict(reader.get_meta_data())
    except Exception:
        pass
    frames: List[np.ndarray] = []
    for frame in reader:
        frames.append(np.asarray(frame)[..., :3])
        if max_frames is not None and len(frames) >= max_frames:
            break
    try:
        reader.close()
    except Exception:
        pass
    if not frames:
        raise IOError("imageio 解码得到 0 帧: %s" % path)
    return np.stack(frames, axis=0), {
        "fps": float(meta.get("fps", 0.0) or 0.0),
        "backend": "imageio",
    }


def decode_video(
    path: str,
    max_frames: Optional[int] = None,
    max_side: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """解码为 uint8 [T,H,W,3] (RGB)。max_side 给定时长边缩放到该值。"""
    errors: List[str] = []
    frames = None
    meta: Dict[str, Any] = {}
    for fn in (_read_cv2, _read_imageio):
        try:
            frames, meta = fn(path, max_frames=max_frames)
            break
        except Exception as exc:  # noqa: BLE001
            errors.append("%s: %s" % (fn.__name__, exc))
    if frames is None:
        raise IOError("所有解码后端均失败 %s | %s" % (path, " ; ".join(errors)))

    if max_side is not None:
        frames = resize_frames(frames, max_side)
    meta["n_frames"] = int(frames.shape[0])
    meta["height"] = int(frames.shape[1])
    meta["width"] = int(frames.shape[2])
    return frames, meta


def resize_frames(frames: np.ndarray, max_side: int) -> np.ndarray:
    """按长边等比缩放（INTER_AREA）。同一次对比里所有视频必须用同一个 max_side。"""
    import cv2

    h, w = int(frames.shape[1]), int(frames.shape[2])
    long_side = max(h, w)
    if long_side <= max_side:
        return frames
    scale = float(max_side) / float(long_side)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return np.stack(
        [cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames], axis=0
    )


def to_gray_float(frames: np.ndarray) -> np.ndarray:
    """uint8 [T,H,W,3] -> float32 [T,H,W] in [0,1]。"""
    import cv2

    arr = frames.astype(np.float32) / 255.0
    out = np.empty(arr.shape[:3], dtype=np.float32)
    for t in range(arr.shape[0]):
        out[t] = cv2.cvtColor(arr[t], cv2.COLOR_RGB2GRAY)
    return out


def to_rgb_float(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float32) / 255.0


def probe_video(path: str, count_frames: bool = True) -> Dict[str, Any]:
    """探测视频元信息；count_frames=True 时真正解码一遍（慢但可靠）。"""
    info: Dict[str, Any] = {"path": os.path.abspath(path)}
    if not os.path.isfile(path):
        info["exists"] = False
        return info

    st = os.stat(path)
    info["exists"] = True
    info["nbytes"] = int(st.st_size)
    info["mtime_ts"] = float(st.st_mtime)
    info["mtime"] = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    info["sha256_16"] = sha256_file(path)[:16]

    try:
        import cv2

        cap = cv2.VideoCapture(path)
        info["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        info["n_frames_header"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        info["decodable"] = True
    except Exception as exc:  # noqa: BLE001
        info["decodable"] = False
        info["decode_error"] = str(exc)

    if count_frames and info.get("decodable"):
        try:
            frames, _ = decode_video(path)
            info["n_frames"] = int(frames.shape[0])
            info["height"] = int(frames.shape[1])
            info["width"] = int(frames.shape[2])
            info["frame_mean"] = float(frames.mean())
        except Exception as exc:  # noqa: BLE001
            info["decodable"] = False
            info["decode_error"] = str(exc)
    return info


def check_run(
    run: Dict[str, Any],
    expected: Optional[Dict[str, Any]] = None,
    run_started_ts: Optional[float] = None,
    video_key: str = "video",
    frame_tolerance: int = 1,
) -> List[str]:
    """校验单个 run 的视频文件，返回问题列表（空列表 = 通过）。"""
    expected = expected or {}
    problems: List[str] = []
    path = run.get(video_key)
    tag = "%s/%s/seed=%s" % (run.get("prompt_id"), run.get("tag"), run.get("seed"))

    if not path:
        return ["%s: manifest 缺少 %s 字段" % (tag, video_key)]
    if not os.path.isfile(path):
        return ["%s: 文件不存在 -> %s（极可能是 cache_video 静默失败）" % (tag, path)]

    info = probe_video(path, count_frames=True)
    if not info.get("decodable"):
        problems.append("%s: 无法解码 -> %s" % (tag, info.get("decode_error")))
        return problems

    if run_started_ts is not None and info["mtime_ts"] < run_started_ts - 5:
        problems.append(
            "%s: 修改时间(%s)早于实验开始时间 -> 疑似复用旧文件 / cache 命中"
            % (tag, info["mtime"])
        )

    exp_frames = expected.get("frames")
    if exp_frames and abs(info["n_frames"] - exp_frames) > frame_tolerance:
        problems.append("%s: 帧数 %d != 期望 %d" % (tag, info["n_frames"], exp_frames))
    exp_w, exp_h = expected.get("width"), expected.get("height")
    if exp_w and info["width"] != exp_w:
        problems.append("%s: 宽度 %d != 期望 %d" % (tag, info["width"], exp_w))
    if exp_h and info["height"] != exp_h:
        problems.append("%s: 高度 %d != 期望 %d" % (tag, info["height"], exp_h))
    exp_fps = expected.get("fps")
    if exp_fps and abs(info.get("fps", 0.0) - exp_fps) > 0.5:
        problems.append("%s: fps %.3f != 期望 %.3f" % (tag, info.get("fps", 0.0), exp_fps))

    run["_video_info"] = info
    return problems


def verify_manifest(
    runs: Sequence[Dict[str, Any]],
    expected: Optional[Dict[str, Any]] = None,
    run_started_ts: Optional[float] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """批量校验；并检查内容 hash 重复（同一文件被多个 run 复用）。"""
    problems: List[str] = []
    infos: List[Dict[str, Any]] = []
    seen: Dict[str, List[str]] = {}

    for run in runs:
        problems.extend(check_run(run, expected=expected, run_started_ts=run_started_ts))
        info = run.get("_video_info")
        if info:
            infos.append(info)
            seen.setdefault(info.get("sha256_16", ""), []).append(str(run.get("prompt_id")))

    for sha, prompts in seen.items():
        if len(prompts) > 1:
            problems.append(
                "内容重复: %d 个 run 的 sha256 相同 (%s) -> 落盘被覆盖或复用了缓存: %s"
                % (len(prompts), sha, ", ".join(sorted(set(prompts))))
            )
    return problems, infos
