"""Train DigitCNN on MNIST and save the best validation checkpoint."""

import argparse
import os
from pathlib import Path

import certifi
import matplotlib
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import DigitCNN

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


def save_learning_curves(history: dict[str, list[float]], output: Path) -> None:
    """Save training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = max(
        range(len(history["validation_accuracy"])),
        key=history["validation_accuracy"].__getitem__,
    ) + 1
    tick_step = max(1, len(history["train_loss"]) // 6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("MNIST CNN Learning Curves", fontsize=18, fontweight="bold")

    axes[0].plot(
        epochs,
        history["train_loss"],
        marker="o",
        linewidth=2.5,
        label="Training",
    )
    axes[0].plot(
        epochs,
        history["validation_loss"],
        marker="s",
        linewidth=2.5,
        label="Validation",
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        marker="o",
        linewidth=2.5,
        label="Training",
    )
    axes[1].plot(
        epochs,
        history["validation_accuracy"],
        marker="s",
        linewidth=2.5,
        label="Validation",
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    for axis in axes:
        axis.axvline(
            best_epoch,
            color="0.45",
            linestyle="--",
            linewidth=1.5,
            label=f"Best validation epoch ({best_epoch})",
        )
        axis.xaxis.set_major_locator(MultipleLocator(tick_step))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--validation-size",
        type=int,
        default=5_000,
        help="Number of MNIST training images reserved for validation",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "checkpoints" / "mnist_cnn.pt"
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=PROJECT_DIR / "examples" / "learning_curves.png",
        help="Path for the combined training and validation curves",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    torch.manual_seed(42)
    # Some python.org/macOS installations do not expose the system CA bundle.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    device = choose_device()
    print(f"Using device: {device}")

    # MNIST's published mean and standard deviation.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_train_set = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    if not 1 <= args.validation_size < len(full_train_set):
        parser.error(
            f"--validation-size must be between 1 and {len(full_train_set) - 1}"
        )

    train_size = len(full_train_set) - args.validation_size
    split_generator = torch.Generator().manual_seed(42)
    train_set, validation_set = random_split(
        full_train_set,
        [train_size, args.validation_size],
        generator=split_generator,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False
    )

    print(
        f"Dataset split: {len(train_set):,} train | "
        f"{len(validation_set):,} validation"
    )

    model = DigitCNN().to(device)
    # CrossEntropyLoss expects raw logits and applies log-softmax internally.
    # Do not apply softmax to the model output before computing this loss.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation_accuracy = -1.0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()       # 1. clear old gradients
            logits = model(images)      # 2. forward pass
            loss = criterion(logits, labels)  # 3. measure error
            loss.backward()             # 4. backpropagation
            optimizer.step()            # 5. update parameters

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

        train_loss = running_loss / seen
        train_accuracy = running_correct / seen
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train accuracy {train_accuracy:.2%} | "
            f"validation loss {validation_loss:.4f} | "
            f"validation accuracy {validation_accuracy:.2%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "epoch": epoch,
                    "normalization": {"mean": (0.1307,), "std": (0.3081,)},
                },
                args.output,
            )

    save_learning_curves(history, args.plot_output)
    print(f"Learning curves saved to {args.plot_output}")

    print(
        f"Best model saved to {args.output} "
        f"(validation accuracy {best_validation_accuracy:.2%})"
    )


if __name__ == "__main__":
    main()
