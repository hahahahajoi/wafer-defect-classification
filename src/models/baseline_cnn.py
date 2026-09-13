import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
            )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
            )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
            )

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(
            in_features=64 * 8 * 8,
            out_features=9
            )


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x        


# if __name__ == "__main__":
#     model = BaselineCNN()

#     x = torch.randn(32, 1, 64, 64)

#     output = model(x)

#     print("Input shape:", x.shape)
#     print("Output shape:", output.shape)       