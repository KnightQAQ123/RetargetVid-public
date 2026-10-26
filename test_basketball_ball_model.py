"""Create a 10-second preview using the basketball-specific detector."""
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "input.mp4"
MODEL_PATH = ROOT / "basketball_ball_detector.pt"
PREVIEW_PATH = ROOT / "basketball_ball_model_test_10s.mp4"
CSV_PATH = ROOT / "basketball_ball_model_test_10s.csv"
CONFIDENCE = 0.12
IMAGE_SIZE = 1280
TEST_SECONDS = 10
START_SECONDS = 55


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError("找不到 input.mp4: {}".format(VIDEO_PATH))
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("找不到篮球专用模型: {}".format(MODEL_PATH))

    model = YOLO(str(MODEL_PATH))
    names = model.names
    if isinstance(names, dict):
        class_names = {int(k): str(v) for k, v in names.items()}
    else:
        class_names = {i: str(name) for i, name in enumerate(names)}
    ball_classes = [
        class_id for class_id, name in class_names.items()
        if "ball" in name.lower() or "basket" in name.lower()
    ]
    if not ball_classes and len(class_names) == 1:
        ball_classes = list(class_names.keys())
    if not ball_classes:
        raise RuntimeError(
            "模型类别中没找到篮球类别：{}。请把这行错误发给我。".format(class_names)
        )

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("无法读取 input.mp4 的视频信息")

    start_frame = min(int(round(fps * START_SECONDS)), max(0, frame_count - 1))
    max_frames = min(int(round(fps * TEST_SECONDS)), frame_count - start_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    device = 0 if torch.cuda.is_available() else "cpu"
    print("模型类别：{}".format(class_names))
    print("本次处理视频第 {} 至 {} 秒（{} 帧），使用设备 {}。".format(
        START_SECONDS, START_SECONDS + TEST_SECONDS, max_frames, device
    ))
    print("预览会保存为：{}".format(PREVIEW_PATH))

    writer = cv2.VideoWriter(
        str(PREVIEW_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建预览视频: {}".format(PREVIEW_PATH))

    frame_index = 0
    frames_with_ball = 0
    boxes_found = 0
    try:
        with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["frame", "confidence", "center_x", "center_y", "x1", "y1", "x2", "y2"])

            while frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(
                    source=frame,
                    classes=ball_classes,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    device=device,
                    half=(device == 0),
                    verbose=False,
                )[0]

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    coords = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                    for box, score in zip(coords, scores):
                        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3)
                        cv2.putText(
                            frame,
                            "BASKETBALL {:.2f}".format(float(score)),
                            (max(5, x1), max(28, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 220, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        csv_writer.writerow(
                            [start_frame + frame_index, round(float(score), 4), round(center_x, 1),
                             round(center_y, 1), x1, y1, x2, y2]
                        )
                        boxes_found += 1
                    frames_with_ball += 1
                else:
                    cv2.putText(
                        frame, "NO BALL DETECTED", (30, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA
                    )

                cv2.putText(
                    frame,
                    "TEST FRAME {}/{}".format(frame_index + 1, max_frames),
                    (30, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                frame_index += 1
                if frame_index % 60 == 0:
                    print("已处理 {} / {} 帧；检测到球的帧 {} 帧。".format(
                        frame_index, max_frames, frames_with_ball
                    ))
    finally:
        capture.release()
        writer.release()

    print("完成：{} 帧中有 {} 帧检测到球，共画出 {} 个框。".format(
        frame_index, frames_with_ball, boxes_found
    ))
    print("请打开这个视频检查：{}".format(PREVIEW_PATH))


if __name__ == "__main__":
    main()
