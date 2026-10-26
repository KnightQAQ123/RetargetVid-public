from pathlib import Path
import csv
import torch

# Load the original RetargetVid checkpoint on CPU in the older environment.
_original_torch_load = torch.load


def _load_checkpoint_on_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _original_torch_load(*args, **kwargs)


torch.load = _load_checkpoint_on_cpu

import smartVidCrop


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "input.mp4"
FOCUS_CSV = ROOT / "basketball_focus.csv"
OUTPUT_PATH = ROOT / "outputs" / "cropped_basketball_9_16"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_focus_points(csv_path):
    points = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            if not row.get("focus_x") or not row.get("focus_y"):
                continue
            try:
                frame_index = int(row["frame"])
                points[frame_index] = (
                    float(row["focus_x"]),
                    float(row["focus_y"]),
                )
            except (TypeError, ValueError):
                continue
    return points


if not FOCUS_CSV.is_file():
    raise FileNotFoundError(
        "找不到 basketball_focus.csv。请先运行 detect_basketball_targets.py。"
    )

focus_points = load_focus_points(FOCUS_CSV)
print("已读取 {} 个篮球/球员焦点位置。".format(len(focus_points)))

params = smartVidCrop.sc_init_crop_params(use_best_settings=True)
params["out_ratio"] = "9:16"

smartVidCrop.smart_vid_crop(
    str(VIDEO_PATH),
    CP=params,
    final_vid_fn=str(OUTPUT_PATH),
    save_vid=True,
    copy_sound=False,
    focus_points=focus_points,
)

print("裁剪完成: {}".format(str(OUTPUT_PATH) + ".mp4"))
