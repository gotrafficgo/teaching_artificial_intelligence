"""Lightweight smoke tests for the CNN example."""

import unittest
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from image_utils import prepare_digit
from model import DigitCNN


PROJECT_DIR = Path(__file__).resolve().parent


class DigitCNNTests(unittest.TestCase):
    def test_model_output_shape_and_probabilities(self) -> None:
        model = DigitCNN().eval()
        inputs = torch.zeros(2, 1, 28, 28)

        with torch.inference_mode():
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)

        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(2)))

    def test_preprocessing_returns_expected_shape(self) -> None:
        image = Image.new("L", (80, 100), color=255)
        draw = ImageDraw.Draw(image)
        draw.line((20, 15, 60, 85), fill=0, width=8)

        preview, tensor = prepare_digit(image)

        self.assertEqual(preview.size, (28, 28))
        self.assertEqual(tuple(tensor.shape), (1, 1, 28, 28))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_blank_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "appears blank"):
            prepare_digit(Image.new("L", (28, 28), color=0))

    def test_included_checkpoint_loads(self) -> None:
        checkpoint_path = PROJECT_DIR / "checkpoints" / "mnist_cnn.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = DigitCNN()
        model.load_state_dict(checkpoint["model_state_dict"])


if __name__ == "__main__":
    unittest.main()
