"""Predict one handwritten digit and print all 0-9 probabilities."""

import argparse
from pathlib import Path

from PIL import Image
import torch

from image_utils import prepare_digit
from model import DigitCNN


PROJECT_DIR = Path(__file__).resolve().parent


def probability_bar(probability: float, width: int = 30) -> str:
    filled = round(probability * width)
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize one handwritten digit")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_DIR / "checkpoints" / "mnist_cnn.pt"
    )
    parser.add_argument("--save-preprocessed", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}\nRun: python train.py")

    # Step 1 — 28 x 28 pixel matrix.
    preview, input_tensor = prepare_digit(Image.open(args.image))
    if args.save_preprocessed:
        args.save_preprocessed.parent.mkdir(parents=True, exist_ok=True)
        preview.save(args.save_preprocessed)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = DigitCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        # Steps 2–6 — Conv/ReLU/Pool blocks, Flatten, then Linear.
        logits = model(input_tensor)
        # Step 7 — Softmax: 10 logits -> 10 probabilities.
        probabilities = torch.softmax(logits, dim=1)[0]

    # Step 8 — Prediction: choose the largest probability.
    prediction = int(probabilities.argmax())
    predicted_probability = float(probabilities[prediction])
    print(
        f"\nPrediction: {prediction}  "
        f"predicted probability: {predicted_probability:.2%}\n"
    )
    for digit, probability in enumerate(probabilities.tolist()):
        marker = "<-" if digit == prediction else "  "
        print(f"{digit}: {probability_bar(probability)} {probability:6.2%} {marker}")


if __name__ == "__main__":
    main()
