"""Generate a slightly irregular handwritten 7 for testing the CNN."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "examples" / "test_digit_7.png"
    )
    args = parser.parse_args()

    # Draw large first, then downsample for smooth, handwriting-like edges.
    image = Image.new("L", (280, 280), color=0)
    draw = ImageDraw.Draw(image)
    draw.line([(65, 65), (215, 61), (178, 112), (145, 165), (120, 225)], fill=255, width=28, joint="curve")
    draw.line([(84, 142), (174, 139)], fill=230, width=18)
    image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Generated handwritten digit 7: {args.output}")


if __name__ == "__main__":
    main()
