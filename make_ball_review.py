from pathlib import Path
import csv

import cv2


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "input.mp4"
DETECTIONS_CSV = ROOT / "ball_detections.csv"
REVIEW_PATH = ROOT / "ball_detection_review.mp4"
MIN_BOX_SIDE = 6
MAX_BOX_SIDE = 110
MAX_BOX_ASPECT = 2.0


def load_best_candidates(csv_path):
    best_by_frame = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("detected") != "1":
                continue
            try:
                frame = int(row["frame"])
                score = float(row["confidence"])
                x1 = int(float(row["x1"]))
                y1 = int(float(row["y1"]))
                x2 = int(float(row["x2"]))
                y2 = int(float(row["y2"]))
            except (TypeError, ValueError, KeyError):
                continue

            box_width = abs(x2 - x1)
            box_height = abs(y2 - y1)
            short_side = min(box_width, box_height)
            long_side = max(box_width, box_height)
            if short_side < MIN_BOX_SIDE or long_side > MAX_BOX_SIDE:
                continue
            if long_side / float(max(1, short_side)) > MAX_BOX_ASPECT:
                continue

            old = best_by_frame.get(frame)
            if old is None or score > old[0]:
                best_by_frame[frame] = (score, x1, y1, x2, y2)
    return best_by_frame


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
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("无法读取 input.mp4 的视频信息")

    candidates = load_best_candidates(DETECTIONS_CSV)
    writer = cv2.VideoWriter(
        str(REVIEW_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建检查视频: {}".format(REVIEW_PATH))

    frame_index = 0
    found_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            candidate = candidates.get(frame_index)
            if candidate is None:
                cv2.putText(
                    frame,
                    "NO BALL CANDIDATE",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            else:
                found_count += 1
                score, x1, y1, x2, y2 = candidate
                color = (0, 220, 0) if score >= 0.15 else (0, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = "BALL CANDIDATE {:.2f}".format(score)
                cv2.putText(
                    frame,
                    label,
                    (max(5, x1), max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "FRAME {}/{}".format(frame_index + 1, frame_count),
                (30, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
            if frame_index % 500 == 0:
                print("检查预览已处理 {} / {} 帧。".format(frame_index, frame_count))
    finally:
        capture.release()
        writer.release()

    print(
        "完成：共 {} 帧，{} 帧画出一个最高分候选框。".format(
            frame_index, found_count
        )
    )
    print("检查视频：{}".format(REVIEW_PATH))


if __name__ == "__main__":
    main()
