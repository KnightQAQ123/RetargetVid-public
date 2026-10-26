from pathlib import Path
import csv
import math

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "input.mp4"
CSV_PATH = ROOT / "basketball_focus.csv"
PREVIEW_PATH = ROOT / "basketball_focus_preview.mp4"

# This larger COCO model is more likely to find a small ball than yolo11n.
MODEL_NAME = "yolo11m.pt"
IMAGE_SIZE = 1280
DETECT_CONFIDENCE = 0.08
PLAYER_CONFIDENCE = 0.25
BALL_CONFIDENCE = 0.08
MAX_HOLD_FRAMES = 10


def is_court_floor_patch(frame, box):
    """Keep people whose feet appear to touch the wooden or blue court."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)

    if foot_y < int(height * 0.25) or foot_y >= height - 2:
        return False

    # Sample just below the person's box, where the floor should be visible.
    x0 = max(0, foot_x - 12)
    x_end = min(width, foot_x + 13)
    y0 = max(0, foot_y - 2)
    y_end = min(height, foot_y + 18)
    patch = frame[y0:y_end, x0:x_end]
    if patch.size == 0:
        return False

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    wood = (hue >= 5) & (hue <= 35) & (sat >= 25) & (val >= 65)
    blue_paint = (hue >= 90) & (hue <= 140) & (sat >= 45) & (val >= 35)
    court_fraction = np.mean(wood | blue_paint)
    return court_fraction >= 0.22


def box_distance(px, py, box):
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)


def find_possessor(ball_x, ball_y, players):
    best_player = None
    best_distance = None
    for player in players:
        x1, y1, x2, y2 = player["box"]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        expanded = (
            x1 - box_w * 0.18,
            y1 - box_h * 0.12,
            x2 + box_w * 0.18,
            y2 + box_h * 0.12,
        )
        distance = box_distance(ball_x, ball_y, expanded)
        if best_distance is None or distance < best_distance:
            best_player = player
            best_distance = distance

    if best_player is not None:
        # A ball close to a player's body/hand is treated as possession.
        threshold = max(24.0, (best_player["box"][3] - best_player["box"][1]) * 0.10)
        if best_distance <= threshold:
            return best_player
    return None


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = [int(value) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError("找不到 input.mp4: {}".format(VIDEO_PATH))

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("无法读取 input.mp4 的视频信息")

    writer = cv2.VideoWriter(
        str(PREVIEW_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("无法创建检测预览视频: {}".format(PREVIEW_PATH))

    model = YOLO(MODEL_NAME)
    results = model.track(
        source=str(VIDEO_PATH),
        classes=[0, 32],  # COCO: person and sports ball
        tracker="bytetrack.yaml",
        imgsz=IMAGE_SIZE,
        conf=DETECT_CONFIDENCE,
        device=0,
        stream=True,
        verbose=False,
    )

    last_focus = None
    missing_ball_frames = 0
    frame_index = 0

    print("正在检测球员和篮球，首次运行会下载模型 {}。".format(MODEL_NAME))
    print("观众过滤后的视频和位置表将保存在项目文件夹中。")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame", "focus_x", "focus_y", "mode", "ball_detected", "court_players"]
        )

        try:
            for result in results:
                frame = result.orig_img.copy()
                frame_h, frame_w = frame.shape[:2]
                players = []
                balls = []

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    confidences = boxes.conf.cpu().numpy()
                    ids = boxes.id.cpu().numpy() if boxes.id is not None else None

                    for i, box_arr in enumerate(xyxy):
                        box = tuple(float(v) for v in box_arr)
                        class_id = int(classes[i])
                        confidence = float(confidences[i])
                        track_id = int(ids[i]) if ids is not None else -1
                        center_x = (box[0] + box[2]) / 2.0
                        center_y = (box[1] + box[3]) / 2.0

                        if class_id == 0 and confidence >= PLAYER_CONFIDENCE:
                            if is_court_floor_patch(frame, box):
                                players.append(
                                    {
                                        "box": box,
                                        "x": center_x,
                                        "y": center_y,
                                        "confidence": confidence,
                                        "id": track_id,
                                    }
                                )
                        elif class_id == 32 and confidence >= BALL_CONFIDENCE:
                            # Exclude detections in the very top/bottom overlays.
                            if frame_h * 0.12 <= center_y <= frame_h * 0.99:
                                balls.append(
                                    {
                                        "box": box,
                                        "x": center_x,
                                        "y": center_y,
                                        "confidence": confidence,
                                        "id": track_id,
                                    }
                                )

                focus = None
                mode = "retargetvid"
                ball_detected = 0
                ball = max(balls, key=lambda item: item["confidence"]) if balls else None

                if ball is not None:
                    ball_detected = 1
                    possessor = find_possessor(ball["x"], ball["y"], players)
                    if possessor is not None:
                        focus = (
                            (possessor["x"] + ball["x"]) / 2.0,
                            (possessor["y"] + ball["y"]) / 2.0,
                        )
                        mode = "player+ball"
                    else:
                        focus = (ball["x"], ball["y"])
                        mode = "ball"
                    last_focus = focus
                    missing_ball_frames = 0
                else:
                    missing_ball_frames += 1
                    if players:
                        # When the generic model misses the ball, keep the group
                        # of on-court players in view instead of spectators.
                        focus = (
                            float(np.median([p["x"] for p in players])),
                            float(np.median([p["y"] for p in players])),
                        )
                        mode = "players fallback"
                        last_focus = focus
                    elif last_focus is not None and missing_ball_frames <= MAX_HOLD_FRAMES:
                        focus = last_focus
                        mode = "short hold"

                preview = frame.copy()
                for player in players:
                    label = "player"
                    if player["id"] >= 0:
                        label += " {}".format(player["id"])
                    draw_box(preview, player["box"], label, (40, 220, 40))
                if ball is not None:
                    label = "ball {:.2f}".format(ball["confidence"])
                    if ball["id"] >= 0:
                        label += " {}".format(ball["id"])
                    draw_box(preview, ball["box"], label, (0, 80, 255))
                if focus is not None:
                    fx, fy = int(focus[0]), int(focus[1])
                    cv2.circle(preview, (fx, fy), 10, (0, 255, 255), -1)
                    cv2.putText(
                        preview,
                        mode,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    focus_x = min(1.0, max(0.0, focus[0] / float(frame_w)))
                    focus_y = min(1.0, max(0.0, focus[1] / float(frame_h)))
                    csv_writer.writerow(
                        [
                            frame_index,
                            "{:.8f}".format(focus_x),
                            "{:.8f}".format(focus_y),
                            mode,
                            ball_detected,
                            len(players),
                        ]
                    )
                else:
                    csv_writer.writerow(
                        [frame_index, "", "", "retargetvid", ball_detected, len(players)]
                    )

                writer.write(preview)
                frame_index += 1
                if frame_index % 300 == 0:
                    print("已处理 {} 帧".format(frame_index))
        finally:
            writer.release()

    print("检测完成，共处理 {} 帧。".format(frame_index))
    print("位置表: {}".format(CSV_PATH))
    print("筛选预览: {}".format(PREVIEW_PATH))


if __name__ == "__main__":
    main()
