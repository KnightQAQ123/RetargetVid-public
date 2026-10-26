from pathlib import Path
import csv
import os

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = Path(os.environ.get("RETARGETVID_VIDEO_PATH", str(ROOT / "input.mp4")))
MODEL_PATH = Path(os.environ.get("RETARGETVID_MODEL_PATH", str(ROOT / "runs" / "detect" / "nba_ball_v4" / "weights" / "best.pt")))
PREVIEW_PATH = Path(os.environ.get("RETARGETVID_PREVIEW_PATH", str(ROOT / "ball_detection_preview.mp4")))
CSV_PATH = Path(os.environ.get("RETARGETVID_DETECTIONS_CSV", str(ROOT / "ball_detections.csv")))

# Match the earlier preview's detector input size; cropping still uses the original video.
CONFIDENCE = 0.25


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError("找不到 input.mp4: {}".format(VIDEO_PATH))
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("找不到篮球模型: {}".format(MODEL_PATH))

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("无法读取 input.mp4 的视频信息")

    device = 0 if torch.cuda.is_available() else "cpu"
    image_size = 640
    print("原视频：{} x {}，{} FPS".format(width, height, round(fps, 2)))
    print("只检测篮球；使用 {}，检测尺寸 {}，设备 {}。".format(MODEL_PATH.name, image_size, device))
    print("开始处理，预览保存到 {}，检测记录保存到 {}。".format(PREVIEW_PATH, CSV_PATH))

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(VIDEO_PATH),
        classes=[0],  # 训练集中的篮球类别
        imgsz=image_size,
        conf=CONFIDENCE,
        device=device,
        half=(device == 0),
        stream=True,
        verbose=False,
    )

    writer = cv2.VideoWriter(
        str(PREVIEW_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("无法创建预览视频: {}".format(PREVIEW_PATH))

    frame_count = 0
    frames_with_ball = 0
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame", "detected", "confidence", "center_x", "center_y", "x1", "y1", "x2", "y2"]
        )

        try:
            for frame_count, result in enumerate(results, start=1):
                frame = result.orig_img.copy()
                found = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    coords = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    for box, score, class_id in zip(coords, scores, classes):
                        if class_id != 0:
                            continue
                        x1, y1, x2, y2 = [int(round(value)) for value in box]
                        found.append((float(score), x1, y1, x2, y2))

                if found:
                    frames_with_ball += 1
                    for score, x1, y1, x2, y2 in found:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 3)
                        label = "BALL {:.2f}".format(score)
                        text_y = max(26, y1 - 9)
                        cv2.putText(
                            frame,
                            label,
                            (max(0, x1), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 80, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        csv_writer.writerow(
                            [frame_count - 1, 1, round(score, 4), round((x1 + x2) / 2, 1),
                             round((y1 + y2) / 2, 1), x1, y1, x2, y2]
                        )
                else:
                    csv_writer.writerow([frame_count - 1, 0, "", "", "", "", "", "", ""])

                writer.write(frame)
                if frame_count % 300 == 0:
                    print("已处理 {} 帧，找到球的帧 {} 帧。".format(frame_count, frames_with_ball))
        finally:
            writer.release()

    print("完成：共处理 {} 帧，其中 {} 帧检测到球。".format(frame_count, frames_with_ball))
    print("打开这个文件检查球框：{}".format(PREVIEW_PATH))


if __name__ == "__main__":
    main()
