import torch
import torch.nn.functional as F


def resize_wafer(wafer_map, target_size=64):
    wafer = torch.tensor(
        wafer_map,
        dtype=torch.float32
    )

    # 원본과 같은 비율 보정을 위해
    height, width = wafer.shape
    scale = target_size / max(height, width)

    new_height = round(height * scale)
    new_width = round(width * scale)
    

    wafer = wafer.unsqueeze(0).unsqueeze(0)

    resized = F.interpolate(
        wafer,
        size=(new_height, new_width),
        mode="nearest"
    )
    pad_height = target_size - new_height
    pad_width = target_size - new_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded = F.pad(
        resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0
    )

    return padded.squeeze(0).squeeze(0)

# wafer 하나 가져와서 test
# if __name__ == "__main__":
#     import pandas as pd

#     df = pd.read_pickle("data/WM811K_labeled.pkl")

#     sample = df["waferMap"].iloc[0]

#     print("Original shape:", sample.shape)

#     resized = resize_wafer(sample)

#     print("Resized shape:", resized.shape)