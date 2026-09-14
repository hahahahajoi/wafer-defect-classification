import pandas as pd
import torch

from torch.utils.data import DataLoader
from src.data.dataset import WaferDataset
from torch.utils.data import DataLoader, WeightedRandomSampler


def create_train_loader(batch_size=32):
    train_df = pd.read_pickle("data/WM811K_train.pkl")

    train_dataset = WaferDataset(train_df)
    
    class_counts = train_df["failureType"].value_counts()

    sample_weights = train_df["failureType"].map(lambda label: 1.0 / class_counts[label]).values

    sample_weights = torch.tensor(sample_weights,dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
        )

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,            #shuffle=True,
        generator=generator
    )

    return train_loader

def create_val_loader(batch_size=32):
    val_df = pd.read_pickle("data/WM811K_val.pkl")

    val_dataset = WaferDataset(val_df)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return val_loader


def create_test_loader(batch_size=32):
    test_df = pd.read_pickle("data/WM811K_test.pkl")

    test_dataset = WaferDataset(test_df)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader

# if __name__ == "__main__":
#     train_loader = create_train_loader()

#     wafers, labels = next(iter(train_loader))

#     print("Wafer batch shape:", wafers.shape)
#     print("Label batch shape:", labels.shape)