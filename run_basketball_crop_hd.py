from pathlib import Path
import csv
import os

import cv2
import torch

# The original RetargetVid checkpoint was saved for CUDA. Load it on CPU in
# this older environment, then let RetargetVid process the video as usual.
_original_torch_load = torch.load


def _load_checkpoint_on_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _original_torch_load(*args, **kwargs)


torch.load = _load_checkpoint_on_cpu

import smartVidCrop


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = Path(os.environ.get("RETARGETVID_VIDEO_PATH", str(ROOT / "input.mp4")))
DETECTIONS_CSV = Path(os.environ.get("RETARGETVID_DETECTIONS_CSV", str(ROOT / "ball_detections.csv")))
OUTPUT_PATH = Path(os.environ.get("RETARGETVID_OUTPUT_STEM", str(ROOT / "outputs" / "cropped_basketball_ball_only_9_16")))
MIN_CONFIDENCE = 0.25
MIN_BOX_SIDE = 6
MAX_BOX_SIDE = 110
MAX_BOX_ASPECT = 2.0


def load_best_ball_detections(csv_path):
    best_by_frame = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("detected") != "1":
                continue
            try:
                frame = int(row["frame"])
                confidence = float(row["confidence"])
                x = float(row["center_x"])
                y = float(row["center_y"])
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except (TypeError, ValueError, KeyError):
                continue

            if confidence < MIN_CONFIDENCE:
                continue
            box_width = abs(x2 - x1)
            box_height = abs(y2 - y1)
            short_side = min(box_width, box_height)
            long_side = max(box_width, box_height)
            if short_side < MIN_BOX_SIDE or long_side > MAX_BOX_SIDE:
                continue
            if long_side / max(1.0, short_side) > MAX_BOX_ASPECT:
                continue
            old = best_by_frame.get(frame)
            if old is None or confidence > old[0]:
                best_by_frame[frame] = (confidence, x, y)
    return best_by_frame


def remove_isolated_jumps(points):
    """Discard a single implausible jump when nearby detections agree."""
    anchors = sorted(points)
    bad_frames = set()
    for i in range(1, len(anchors) - 1):
        before = anchors[i - 1]
        current = anchors[i]
        after = anchors[i + 1]
        if current - before > 18 or after - current > 18:
            continue

        x0, y0 = points[before]
        x1, y1 = points[after]
        if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > 240:
            continue

        ratio = float(current - before) / (after - before)
        expected_x = x0 + (x1 - x0) * ratio
        expected_y = y0 + (y1 - y0) * ratio
        actual_x, actual_y = points[current]
        error = ((actual_x - expected_x) ** 2 + (actual_y - expected_y) ** 2) ** 0.5
        if error > 220:
            bad_frames.add(current)

    return {frame: point for frame, point in points.items() if frame not in bad_frames}


def fill_all_gaps(points, frame_count):
    """Interpolate the ball path through the whole video; never use saliency fallback."""
    if not points:
        return {}

    anchors = sorted(points)
    filled = dict(points)
    for left, right in zip(anchors, anchors[1:]):
        if right - left <= 1:
            continue
        x0, y0 = points[left]
        x1, y1 = points[right]
        span = right - left
        for frame in range(left + 1, right):
            ratio = float(frame - left) / span
            filled[frame] = (
                x0 + (x1 - x0) * ratio,
                y0 + (y1 - y0) * ratio,
            )

    # Keep the ball at the nearest known position before the first and after the last detection.
    first = anchors[0]
    for frame in range(0, first):
        filled[frame] = points[first]
    last = anchors[-1]
    for frame in range(last + 1, frame_count):
        filled[frame] = points[last]
    return filled


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError("找不到 input.mp4: {}".format(VIDEO_PATH))
    if not DETECTIONS_CSV.is_file():
        raise FileNotFoundError(
            "找不到 ball_detections.csv。请先运行 find_basketball_hd.py。"
        )

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError("无法读取 input.mp4 的视频信息")

    best = load_best_ball_detections(DETECTIONS_CSV)
    pixel_points = remove_isolated_jumps(
        {frame: (x, y) for frame, (_, x, y) in best.items()}
    )
    focus_points = fill_all_gaps(pixel_points, frame_count)
    focus_points = {
        frame: (x / float(width), y / float(height))
        for frame, (x, y) in focus_points.items()
    }
    if not focus_points:
        raise RuntimeError(
            "没有置信度达到 {:.2f} 的篮球位置；请先确认球检测预览。".format(
                MIN_CONFIDENCE
            )
        )

    print("已读取 {} 帧较可信的篮球位置，过滤了孤立跳点。".format(len(pixel_points)))
    print("所有帧都只沿篮球位置取景；漏检区间按相邻球位补齐，不切回球员或普通取景。")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    params = smartVidCrop.sc_init_crop_params(use_best_settings=True)
    params["out_ratio"] = "9:16"
    # Follow the basketball promptly; the default stability pass can hold the crop
    # away from a moving ball for too long.
    params["focus_stability"] = False
    params["loess_filt"] = 0
    params["lp_filt"] = 1
    params["lp_cutoff"] = 3.0
    params["lp_order"] = 2

    smartVidCrop.smart_vid_crop(
        str(VIDEO_PATH),
        CP=params,
        final_vid_fn=str(OUTPUT_PATH),
        save_vid=True,
        copy_sound=False,
        focus_points=focus_points,
    )

    print("裁剪完成: {}.mp4".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()
