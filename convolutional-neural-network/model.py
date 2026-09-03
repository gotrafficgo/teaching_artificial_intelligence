"""A small CNN for handwritten-digit recognition."""

import torch
from torch import nn


class DigitCNN(nn.Module):
    """Input: [batch, 1, 28, 28]; output: [batch, 10] logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Step 2 — Convolution: detect local patterns.
            # [B, 1, 28, 28] -> [B, 16, 28, 28]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # Step 3 — ReLU: keep positive responses.
            nn.ReLU(),
            # Step 4 — Pooling: reduce the spatial size.
            # [B, 16, 28, 28] -> [B, 16, 14, 14]
            nn.MaxPool2d(kernel_size=2),
            # Step 5 — More convolution: combine simple patterns.
            # [B, 16, 14, 14] -> [B, 32, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # The second convolutional block also uses ReLU and pooling.
            nn.ReLU(),
            # [B, 32, 14, 14] -> [B, 32, 7, 7]
            nn.MaxPool2d(kernel_size=2),
        )
        # Step 6 — Linear layer: produce 10 scores (logits).
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # Flatten: [B, 32, 7, 7] -> [B, 1568]
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    model = DigitCNN()
    example = torch.zeros(1, 1, 28, 28)
    print(model)
    print("input shape: ", tuple(example.shape))
    print("output shape:", tuple(model(example).shape))
    print("parameters:  ", sum(p.numel() for p in model.parameters()))
