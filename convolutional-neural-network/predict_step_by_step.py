"""Run one prediction and display every intermediate tensor shape."""

import argparse
from pathlib import Path

from PIL import Image
import torch

from image_utils import prepare_digit
from model import DigitCNN


PROJECT_DIR = Path(__file__).resolve().parent


def show_step(number: int, title: str, purpose: str, tensor: torch.Tensor) -> None:
    """Print one pipeline step and the resulting tensor shape."""
    print(f"\n{number}. {title}")
    print(f"   {purpose}")
    print(f"   shape: {list(tensor.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one CNN prediction step by step")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_DIR / "checkpoints" / "mnist_cnn.pt"
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}\nRun: python train.py")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = DigitCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Step 1 — The computer receives a normalized 28 x 28 pixel matrix.
    _, x = prepare_digit(Image.open(args.image))
    show_step(
        1,
        "Normalized 28 x 28 pixel matrix",
        "Pixel intensities are standardized with the MNIST mean and standard deviation.",
        x,
    )

    with torch.inference_mode():
        # Step 2 — 16 learned 3 x 3 kernels produce 16 feature maps.
        x = model.features[0](x)
        show_step(2, "Convolution 1", "Detect local patterns with 16 learned kernels.", x)

        # Step 3 — Negative convolution responses become zero.
        x = model.features[1](x)
        show_step(3, "ReLU 1", "Keep positive responses; set negative responses to zero.", x)

        # Step 4 — Each non-overlapping 2 x 2 region keeps its maximum.
        x = model.features[2](x)
        show_step(4, "Max pooling 1", "Reduce each 28 x 28 map to 14 x 14.", x)

        # Step 5 — A complete second-layer kernel combines all 16 channels.
        x = model.features[3](x)
        show_step(5, "Convolution 2", "Combine simple patterns into 32 new feature maps.", x)

        # Steps 6-7 — Complete the second convolutional block.
        x = model.features[4](x)
        show_step(6, "ReLU 2", "Keep positive responses in the 32 maps.", x)
        x = model.features[5](x)
        show_step(7, "Max pooling 2", "Reduce each 14 x 14 map to 7 x 7.", x)

        # Steps 8-9 — Flatten is only a reshape; Linear produces the logits.
        x = torch.flatten(x, start_dim=1)
        show_step(8, "Flatten", "Reshape 32 x 7 x 7 into 1,568 values.", x)
        logits = model.classifier(x)
        show_step(9, "Linear layer", "Combine 1,568 features to produce 10 logits.", logits)

        # Step 10 — Softmax has no trainable parameters.
        probabilities = torch.softmax(logits, dim=1)
        show_step(
            10,
            "Softmax",
            "Convert the 10 logits into probabilities that sum to 1.",
            probabilities,
        )

        # Step 11 — Argmax returns the index of the largest probability.
        prediction = int(probabilities.argmax(dim=1).item())
        predicted_probability = float(probabilities[0, prediction].item())

    print("\n11. Prediction")
    print("   Select the class with the highest probability (argmax).")
    print(f"   predicted digit: {prediction}")
    print(f"   predicted probability: {predicted_probability:.2%}")
    print("   Note: a Softmax probability is not a calibrated confidence score.")


if __name__ == "__main__":
    main()
