"""Convert a normal handwritten image to MNIST-like model input."""

from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TF


def prepare_digit(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
    """Return the 28x28 preview and normalized [1, 1, 28, 28] tensor.

    A simple corner-brightness heuristic handles typical black-on-white paper
    images and white-on-black MNIST-style images. The foreground is cropped,
    resized into a 20x20 box, and geometrically centered.
    """
    image = ImageOps.grayscale(image)

    # A bright corner normally means dark ink on white paper: invert it.
    corners = [image.getpixel((x, y)) for x in (0, image.width - 1) for y in (0, image.height - 1)]
    if sum(corners) / len(corners) > 127:
        image = ImageOps.invert(image)

    image = ImageOps.autocontrast(image)
    # Ignore very faint noise while finding the handwritten digit's bounds.
    mask = image.point(lambda p: 255 if p > 20 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("The image appears blank; please provide a handwritten digit.")

    digit = image.crop(bbox)
    scale = min(20 / digit.width, 20 / digit.height)
    new_size = (max(1, round(digit.width * scale)), max(1, round(digit.height * scale)))
    digit = digit.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - digit.width) // 2, (28 - digit.height) // 2)
    canvas.paste(digit, offset)

    tensor = TF.to_tensor(canvas)
    tensor = TF.normalize(tensor, mean=(0.1307,), std=(0.3081,))
    return canvas, tensor.unsqueeze(0)
