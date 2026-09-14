import torch

from torch.utils.data import Dataset
from src.preprocessing.resize_wafer import resize_wafer

class WaferDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.label_to_idx = {
        "none": 0,
        "Center": 1,
        "Donut": 2,
        "Edge-Loc": 3,
        "Edge-Ring": 4,
        "Loc": 5,
        "Near-full": 6,
        "Random": 7,
        "Scratch": 8
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        wafer_map = row["waferMap"]
        label = row["failureType"]

        wafer = resize_wafer(wafer_map)
        wafer = wafer.unsqueeze(0)
        label = self.label_to_idx[label]

        return wafer, label