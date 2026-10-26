from pathlib import Path
import torch

# 把显卡保存的模型权重转到 CPU 上读取
_original_torch_load = torch.load

def _load_checkpoint_on_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _original_torch_load(*args, **kwargs)

torch.load = _load_checkpoint_on_cpu

import smartVidCrop

root = Path(__file__).resolve().parent
video = root / "input.mp4"
output = root / "outputs" / "cropped_9_16"
output.parent.mkdir(parents=True, exist_ok=True)

params = smartVidCrop.sc_init_crop_params(use_best_settings=True)
params["out_ratio"] = "9:16"

smartVidCrop.smart_vid_crop(
    str(video),
    CP=params,
    final_vid_fn=str(output),
    save_vid=True,
    copy_sound=False,
)