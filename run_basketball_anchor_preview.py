from pathlib import Path
import csv
import math
import os
import sys

import cv2
import numpy as np
import torch

# RetargetVid 原模型的权重保存为 CUDA 格式；这个脚本在旧的 CPU 环境里加载。
_original_torch_load = torch.load


def _load_checkpoint_on_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _original_torch_load(*args, **kwargs)


torch.load = _load_checkpoint_on_cpu
# SciPy 1.7 在 Windows 上会 chdir 到 DLL 目录；此运行环境禁止该切换。
# 把 DLL 目录加入 PATH，并让 SciPy 使用绝对路径加载 DLL。
_scipy_dll_dir = Path(sys.prefix) / "Lib" / "site-packages" / "scipy" / ".libs"
if _scipy_dll_dir.is_dir():
    os.environ["PATH"] = str(_scipy_dll_dir) + os.pathsep + os.environ.get("PATH", "")
_original_chdir = os.chdir

def _safe_scipy_chdir(target):
    if os.path.normcase(os.path.abspath(target)) == os.path.normcase(str(_scipy_dll_dir)):
        return
    _original_chdir(target)

os.chdir = _safe_scipy_chdir
try:
    import smartVidCrop
finally:
    os.chdir = _original_chdir


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "input.mp4"
DETECTIONS_CSV = ROOT / "ball_detections.csv"
OUTPUT_DIR = ROOT / "outputs"
TEMP_VIDEO = OUTPUT_DIR / "_basketball_anchor_source.mp4"
TRACK_CSV = OUTPUT_DIR / "basketball_anchor_track.csv"
REVIEW_VIDEO = OUTPUT_DIR / "basketball_anchor_review.mp4"
CROP_OUTPUT = OUTPUT_DIR / "basketball_anchor_preview_9_16"

# 用户圈出的球对应约 98 秒；检测表中该球在第 2646 帧有一个低置信度框。
ANCHOR_FRAME = 2646
ANCHOR_X = 1234.0
ANCHOR_Y = 733.0
START_SECONDS = 97.3
END_SECONDS = 98.8

MIN_CONFIDENCE = 0.01
MIN_BOX_SIDE = 6
MAX_BOX_SIDE = 70
MAX_BOX_ASPECT = 1.7
MAX_SPEED_PER_FRAME = 75.0
MAX_EXTRAPOLATE_FRAMES = 4
MAX_CONTIGUOUS_GAP = 4
SMOOTH_WINDOW = 7


def load_candidates(csv_path):
    by_frame = {}
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

            box_w = abs(x2 - x1)
            box_h = abs(y2 - y1)
            short_side = min(box_w, box_h)
            long_side = max(box_w, box_h)
            if confidence < MIN_CONFIDENCE:
                continue
            if short_side < MIN_BOX_SIDE or long_side > MAX_BOX_SIDE:
                continue
            if long_side / max(1.0, short_side) > MAX_BOX_ASPECT:
                continue

            candidate = {
                "frame": frame,
                "x": x,
                "y": y,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "width": box_w,
                "height": box_h,
            }
            by_frame.setdefault(frame, []).append(candidate)

    # 同一帧内去掉几乎重叠的重复框。
    for frame, candidates in by_frame.items():
        ordered = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
        kept = []
        for candidate in ordered:
            if all(
                math.hypot(candidate["x"] - other["x"], candidate["y"] - other["y"]) > 10
                for other in kept
            ):
                kept.append(candidate)
        by_frame[frame] = kept
    return by_frame


def choose_seed(by_frame):
    candidates = by_frame.get(ANCHOR_FRAME, [])
    if not candidates:
        raise RuntimeError(
            "第 {} 帧没有球候选框。请确认 ball_detections.csv 是最新生成的。".format(
                ANCHOR_FRAME
            )
        )
    seed = min(
        candidates,
        key=lambda item: math.hypot(item["x"] - ANCHOR_X, item["y"] - ANCHOR_Y),
    )
    distance = math.hypot(seed["x"] - ANCHOR_X, seed["y"] - ANCHOR_Y)
    if distance > 45:
        raise RuntimeError(
            "第 {} 帧最近的候选框离圈出的球太远（{:.0f} 像素）；停止裁剪以免跟错。".format(
                ANCHOR_FRAME, distance
            )
        )
    return seed


def walk_from_seed(by_frame, seed, frame_numbers):
    accepted = {seed["frame"]: seed}
    walk_frames = list(frame_numbers)
    direction = 1 if walk_frames and walk_frames[0] > seed["frame"] else -1
    observations = [seed]

    for frame in walk_frames:
        last = observations[-1]
        elapsed = abs(frame - last["frame"])
        if elapsed > MAX_CONTIGUOUS_GAP:
            break
        if elapsed == 0:
            continue

        candidates = by_frame.get(frame, [])
        if not candidates:
            continue

        predicted_x = last["x"]
        predicted_y = last["y"]
        if len(observations) >= 2:
            previous = observations[-2]
            frame_delta = last["frame"] - previous["frame"]
            if frame_delta != 0:
                vx = (last["x"] - previous["x"]) / float(frame_delta)
                vy = (last["y"] - previous["y"]) / float(frame_delta)
                speed = math.hypot(vx, vy)
                if speed > MAX_SPEED_PER_FRAME:
                    scale = MAX_SPEED_PER_FRAME / speed
                    vx *= scale
                    vy *= scale
                predicted_x += vx * (frame - last["frame"])
                predicted_y += vy * (frame - last["frame"])

        max_distance = MAX_SPEED_PER_FRAME * elapsed + 24.0
        ranked = []
        for candidate in candidates:
            distance = math.hypot(candidate["x"] - predicted_x, candidate["y"] - predicted_y)
            if distance > max_distance:
                continue

            # 连续位置优先；检测分数作为较轻的辅助项。
            score = distance + 24.0 * (1.0 - min(1.0, candidate["confidence"]))
            ranked.append((score, candidate))

        if not ranked:
            continue

        candidate = min(ranked, key=lambda pair: pair[0])[1]
        accepted[frame] = candidate
        observations.append(candidate)

    return accepted


def remove_isolated_jumps(observations):
    frames = sorted(observations)
    bad = set()
    for i in range(1, len(frames) - 1):
        before, current, after = frames[i - 1], frames[i], frames[i + 1]
        if current == ANCHOR_FRAME:
            continue
        if current - before > 18 or after - current > 18:
            continue
        p0 = observations[before]
        p1 = observations[current]
        p2 = observations[after]
        span = after - before
        ratio = (current - before) / float(span)
        expected_x = p0["x"] + (p2["x"] - p0["x"]) * ratio
        expected_y = p0["y"] + (p2["y"] - p0["y"]) * ratio
        error = math.hypot(p1["x"] - expected_x, p1["y"] - expected_y)
        if error > 120:
            bad.add(current)
    return {frame: item for frame, item in observations.items() if frame not in bad}


def make_dense_track(observations, start_frame, end_frame):
    frames = sorted(observations)
    if not frames:
        raise RuntimeError("没有跟到篮球候选框。")
    points = {}
    for left, right in zip(frames, frames[1:]):
        p0 = observations[left]
        p1 = observations[right]
        gap = right - left
        for frame in range(left, right + 1):
            ratio = (frame - left) / float(gap)
            points[frame] = (
                p0["x"] + (p1["x"] - p0["x"]) * ratio,
                p0["y"] + (p1["y"] - p0["y"]) * ratio,
            )
    first = frames[0]
    last = frames[-1]
    for frame in range(start_frame, first):
        points[frame] = (observations[first]["x"], observations[first]["y"])
    for frame in range(last + 1, end_frame + 1):
        points[frame] = (observations[last]["x"], observations[last]["y"])

    # 检测短暂漏帧时用相邻点插值；很长的漏检区间不外推成虚假的快速移动。
    for frame in range(start_frame, end_frame + 1):
        if frame in observations:
            continue
        nearest = min(frames, key=lambda observed: abs(observed - frame))
        if abs(nearest - frame) > MAX_EXTRAPOLATE_FRAMES:
            points[frame] = (observations[nearest]["x"], observations[nearest]["y"])

    raw_frames = np.arange(start_frame, end_frame + 1)
    xs = np.array([points[int(frame)][0] for frame in raw_frames], dtype=np.float32)
    ys = np.array([points[int(frame)][1] for frame in raw_frames], dtype=np.float32)
    kernel = np.ones(SMOOTH_WINDOW, dtype=np.float32) / float(SMOOTH_WINDOW)
    pad = SMOOTH_WINDOW // 2
    xs = np.convolve(np.pad(xs, (pad, pad), mode="edge"), kernel, mode="valid")
    ys = np.convolve(np.pad(ys, (pad, pad), mode="edge"), kernel, mode="valid")
    return {
        int(frame): (float(xs[i]), float(ys[i]))
        for i, frame in enumerate(raw_frames)
    }


def draw_label(frame, text, position, color):
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
    )


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError("找不到 input.mp4: {}".format(VIDEO_PATH))
    if not DETECTIONS_CSV.is_file():
        raise FileNotFoundError(
            "找不到 ball_detections.csv。请先运行 find_basketball_hd.py。"
        )

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
        capture.release()
        raise RuntimeError("无法读取 input.mp4 的视频信息")

    start_frame = max(0, int(round(START_SECONDS * fps)))
    end_frame = min(total_frames - 1, int(round(END_SECONDS * fps)))
    if not (start_frame <= ANCHOR_FRAME <= end_frame):
        capture.release()
        raise RuntimeError("预览时间范围没有包含篮球锚点帧。")

    by_frame = load_candidates(DETECTIONS_CSV)
    seed = choose_seed(by_frame)

    forward = walk_from_seed(by_frame, seed, range(ANCHOR_FRAME + 1, end_frame + 1))
    backward = walk_from_seed(by_frame, seed, range(ANCHOR_FRAME - 1, start_frame - 1, -1))
    observations = dict(backward)
    observations.update(forward)
    observations = remove_isolated_jumps(observations)
    track = make_dense_track(observations, start_frame, end_frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer_temp = cv2.VideoWriter(
        str(TEMP_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    writer_review = cv2.VideoWriter(
        str(REVIEW_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer_temp.isOpened() or not writer_review.isOpened():
        capture.release()
        writer_temp.release()
        writer_review.release()
        raise RuntimeError("无法创建篮球预览视频。")

    focus_points = {}
    with open(TRACK_CSV, "w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["clip_frame", "source_frame", "focus_x", "focus_y", "detected", "confidence"]
        )
        source_frame = 0
        written = 0
        try:
            while source_frame <= end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if source_frame < start_frame:
                    source_frame += 1
                    continue

                x, y = track[source_frame]
                local_frame = source_frame - start_frame
                focus_points[local_frame] = (x / float(width), y / float(height))
                candidate = observations.get(source_frame)
                writer_temp.write(frame.copy())
                if candidate is None:
                    confidence = ""
                    detected = 0
                else:
                    confidence = "{:.4f}".format(candidate["confidence"])
                    detected = 1
                    x1, y1, x2, y2 = [int(round(value)) for value in candidate["box"]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
                    draw_label(
                        frame,
                        "ball candidate {:.2f}".format(candidate["confidence"]),
                        (max(4, x1), max(24, y1 - 7)),
                        (0, 220, 80),
                    )

                cx, cy = int(round(x)), int(round(y))
                cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 2)
                cv2.drawMarker(
                    frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 22, 2
                )
                draw_label(frame, "basketball focus", (20, 40), (0, 255, 255))

                writer_review.write(frame)
                csv_writer.writerow(
                    [
                        local_frame,
                        source_frame,
                        "{:.7f}".format(x / float(width)),
                        "{:.7f}".format(y / float(height)),
                        detected,
                        confidence,
                    ]
                )
                written += 1
                source_frame += 1
        finally:
            capture.release()
            writer_temp.release()
            writer_review.release()

    if written != end_frame - start_frame + 1:
        raise RuntimeError("预览片段帧数不完整，已停止裁剪。")

    params = smartVidCrop.sc_init_crop_params(use_best_settings=True)
    params["out_ratio"] = "9:16"
    params["focus_stability"] = False
    params["loess_filt"] = 0
    params["lp_filt"] = 1
    params["lp_cutoff"] = 3.0
    params["lp_order"] = 2

    smartVidCrop.smart_vid_crop(
        str(TEMP_VIDEO),
        CP=params,
        final_vid_fn=str(CROP_OUTPUT),
        save_vid=True,
        copy_sound=False,
        focus_points=focus_points,
    )

    print("锚点：第 {} 帧，位置 ({:.0f}, {:.0f})，候选分数 {:.3f}".format(
        seed["frame"], seed["x"], seed["y"], seed["confidence"]
    ))
    print("本次预览：{}–{} 秒，共接受 {} 个连续球候选框。".format(
        START_SECONDS, END_SECONDS, len(observations)
    ))
    print("带框检查视频：{}".format(REVIEW_VIDEO))
    print("篮球主体竖屏预览：{}.mp4".format(CROP_OUTPUT))
    print("轨迹位置表：{}".format(TRACK_CSV))


if __name__ == "__main__":
    main()







