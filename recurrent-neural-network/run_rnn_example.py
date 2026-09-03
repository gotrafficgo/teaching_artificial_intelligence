"""Train an RNN, select its best checkpoint, and generate text."""

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RNN_PROGRAM = PROJECT_DIR / "rnn_text_generation.py"


def run(command: list[str], dry_run: bool) -> None:
    """Print a command and run it with the active Python environment."""
    print(f"\n$ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the character-level RNN and generate text"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--prompt", default="the moon moves")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running them"
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.length < 1:
        parser.error("--length must be at least 1")
    if not args.prompt:
        parser.error("--prompt cannot be empty")
    if not args.dry_run and importlib.util.find_spec("torch") is None:
        raise SystemExit(
            "PyTorch is not installed. Activate the shared environment and run: "
            "python -m pip install -r requirements.txt from the repository root."
        )

    python = sys.executable
    checkpoint = PROJECT_DIR / "checkpoints" / "char_rnn.pt"
    print(f"RNN project: {PROJECT_DIR}")
    print(f"Python:      {python}")

    run(
        [
            python,
            str(RNN_PROGRAM),
            "train",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--output",
            str(checkpoint),
        ],
        args.dry_run,
    )
    run(
        [
            python,
            str(RNN_PROGRAM),
            "generate",
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            args.prompt,
            "--length",
            str(args.length),
            "--device",
            args.device,
        ],
        args.dry_run,
    )

    if args.dry_run:
        print("\nDry run completed; the RNN was not executed.")
    else:
        print("\nRNN training and generation completed successfully.")


if __name__ == "__main__":
    main()
