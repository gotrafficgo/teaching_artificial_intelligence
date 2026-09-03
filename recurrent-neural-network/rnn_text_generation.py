"""Train a character-level RNN and use it to generate English text."""

import argparse
import random
from pathlib import Path

import matplotlib
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = PROJECT_DIR / "data" / "alice_in_wonderland.txt"


def choose_device(requested: str) -> torch.device:
    """Select a device, or validate the device requested on the command line."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CharacterDataset(Dataset):
    """Create overlapping input/target sequences from encoded text.

    For the text ``hello`` and a sequence length of 3, one sample is:
        input  = h, e, l
        target = e, l, l
    The target is always the input shifted one character to the left.
    """

    def __init__(
        self, encoded_text: torch.Tensor, sequence_length: int, stride: int = 1
    ) -> None:
        if encoded_text.numel() <= sequence_length:
            raise ValueError("The corpus must be longer than the sequence length.")
        self.encoded_text = encoded_text
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self) -> int:
        available_starts = self.encoded_text.numel() - self.sequence_length
        return (available_starts - 1) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        input_sequence = self.encoded_text[start : start + self.sequence_length]
        target_sequence = self.encoded_text[
            start + 1 : start + self.sequence_length + 1
        ]
        return input_sequence, target_sequence


class CharacterRNN(nn.Module):
    """Embedding -> vanilla RNN -> linear output layer."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, tokens: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tokens)
        rnn_output, hidden = self.rnn(embedded, hidden)
        logits = self.output(rnn_output)
        return logits, hidden


def load_corpus(text_file: Path | None, repeats: int) -> str:
    """Load the included corpus or a user-provided text file."""
    corpus_path = text_file if text_file is not None else DEFAULT_CORPUS_PATH
    text = corpus_path.read_text(encoding="utf-8").lower().strip()
    if repeats > 1:
        text = ((text + "\n") * repeats).strip()
    if not text:
        raise ValueError("The training corpus is empty.")
    return text


def build_vocabulary(text: str) -> tuple[list[str], dict[str, int]]:
    characters = sorted(set(text))
    return characters, {character: index for index, character in enumerate(characters)}


def split_corpus(
    text: str, validation_fraction: float, seed: int
) -> tuple[str, str]:
    """Split complete lines so no sentence appears in both datasets."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        split_index = int(len(text) * (1.0 - validation_fraction))
        return text[:split_index], text[split_index:]

    random.Random(seed).shuffle(lines)
    validation_line_count = max(1, round(len(lines) * validation_fraction))
    validation_lines = lines[:validation_line_count]
    training_lines = lines[validation_line_count:]
    return "\n".join(training_lines), "\n".join(validation_lines)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    vocabulary_size: int,
    device: torch.device,
) -> tuple[float, float]:

    """Calculate next-character loss and accuracy without updating the model."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    token_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, _ = model(inputs)
        loss = criterion(
            logits.reshape(-1, vocabulary_size), targets.reshape(-1)
        )
        batch_tokens = targets.numel()
        loss_sum += loss.item() * batch_tokens
        correct += (logits.argmax(dim=2) == targets).sum().item()
        token_count += batch_tokens
    return loss_sum / token_count, correct / token_count


def save_learning_curves(history: dict[str, list[float]], output: Path) -> int:
    """Save training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = min(
        range(len(history["validation_loss"])),
        key=history["validation_loss"].__getitem__,
    ) + 1
    tick_step = max(1, len(history["train_loss"]) // 6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Character RNN Learning Curves", fontsize=18, fontweight="bold")

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
    axes[1].set(title="Next-Character Accuracy", xlabel="Epoch", ylabel="Accuracy")
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
    return best_epoch


@torch.no_grad()
def generate_text(
    model: CharacterRNN,
    prompt: str,
    length: int,
    char_to_index: dict[str, int],
    index_to_char: list[str],
    device: torch.device,
    temperature: float = 0.8,
    top_k: int = 0,
) -> str:
    """Generate characters autoregressively from a starting prompt."""
    model.eval()
    prompt = prompt.lower()
    if not prompt:
        raise ValueError("The generation prompt cannot be empty.")
    unknown = sorted(set(prompt) - set(char_to_index))
    if unknown:
        readable = ", ".join(repr(character) for character in unknown)
        raise ValueError(f"Prompt contains characters outside the vocabulary: {readable}")

    # Feeding the entire prompt creates a hidden state containing its context.
    prompt_tokens = torch.tensor(
        [[char_to_index[character] for character in prompt]],
        dtype=torch.long,
        device=device,
    )
    logits, hidden = model(prompt_tokens)
    next_logits = logits[:, -1, :]
    result = list(prompt)

    for _ in range(length):
        scaled_logits = next_logits.squeeze(0) / temperature
        if 0 < top_k < scaled_logits.numel():
            values, indices = torch.topk(scaled_logits, top_k)
            probabilities = torch.softmax(values, dim=0)
            sampled_position = torch.multinomial(probabilities, 1).item()
            next_index = indices[sampled_position].item()
        else:
            probabilities = torch.softmax(scaled_logits, dim=0)
            next_index = torch.multinomial(probabilities, 1).item()

        next_character = index_to_char[next_index]
        result.append(next_character)
        next_token = torch.tensor([[next_index]], dtype=torch.long, device=device)
        logits, hidden = model(next_token, hidden)
        next_logits = logits[:, -1, :]

    return "".join(result)


def save_checkpoint(
    output: Path,
    model: CharacterRNN,
    characters: list[str],
    embedding_size: int,
    hidden_size: int,
    num_layers: int,
    epoch: int,
    validation_loss: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "characters": characters,
            "embedding_size": embedding_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epoch": epoch,
            "validation_loss": validation_loss,
        },
        output,
    )


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[CharacterRNN, list[str], dict[str, int], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    characters = checkpoint["characters"]
    model = CharacterRNN(
        vocab_size=len(characters),
        embedding_size=checkpoint["embedding_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    char_to_index = {character: index for index, character in enumerate(characters)}
    return model, characters, char_to_index, checkpoint


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    text = load_corpus(args.text_file, args.corpus_repeats)
    characters, char_to_index = build_vocabulary(text)
    training_text, validation_text = split_corpus(
        text, args.validation_fraction, args.seed
    )
    encoded_training_text = torch.tensor(
        [char_to_index[character] for character in training_text], dtype=torch.long
    )
    encoded_validation_text = torch.tensor(
        [char_to_index[character] for character in validation_text], dtype=torch.long
    )
    minimum_segment_length = args.sequence_length + 1
    if (
        encoded_training_text.numel() < minimum_segment_length
        or encoded_validation_text.numel() < minimum_segment_length
    ):
        raise ValueError(
            "The corpus is too short for this sequence length and validation split. "
            "Use a longer corpus or a shorter sequence length."
        )
    training_dataset = CharacterDataset(
        encoded_training_text, args.sequence_length, args.sequence_stride
    )
    validation_dataset = CharacterDataset(
        encoded_validation_text, args.sequence_length, args.sequence_stride
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = CharacterRNN(
        vocab_size=len(characters),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Using device: {device}")
    print(f"Corpus length: {len(text):,} characters")
    print(f"Vocabulary size: {len(characters)} characters")
    print(f"Training sequences: {len(training_dataset):,}")
    print(f"Validation sequences: {len(validation_dataset):,}")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        token_count = 0

        for inputs, targets in training_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()                 # 1. Clear old gradients.
            logits, _ = model(inputs)             # 2. Run the RNN forward.
            loss = criterion(                     # 3. Compare every prediction.
                logits.reshape(-1, len(characters)),
                targets.reshape(-1),
            )
            loss.backward()                       # 4. Backpropagate through time.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()                      # 5. Update all parameters.

            batch_tokens = targets.numel()
            loss_sum += loss.item() * batch_tokens
            correct += (logits.argmax(dim=2) == targets).sum().item()
            token_count += batch_tokens

        training_loss = loss_sum / token_count
        training_accuracy = correct / token_count

        validation_loss, validation_accuracy = evaluate(
            model,             # the model that has just completed one training epoch
            validation_loader, # the validation mini-batches
            criterion,         # the cross-entropy loss function
            len(characters),   # the vocabulary size, currently 49
            device,            # the computing device used by the model
        )
        history["train_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(training_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"training loss {training_loss:.4f} | "
            f"training accuracy {training_accuracy:.2%} | "
            f"validation loss {validation_loss:.4f} | "
            f"validation accuracy {validation_accuracy:.2%}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                args.output,
                model,
                characters,
                args.embedding_size,
                args.hidden_size,
                args.num_layers,
                epoch,
                validation_loss,
            )

    best_epoch = save_learning_curves(history, args.plot_output)
    print(f"Learning curve saved to {args.plot_output}")
    print(
        f"Best checkpoint: epoch {best_epoch} | "
        f"validation loss {best_validation_loss:.4f} | {args.output}"
    )


def predict(args: argparse.Namespace) -> None:
    if args.length < 1:
        raise ValueError("--length must be at least 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    model, characters, char_to_index, checkpoint = load_checkpoint(
        args.checkpoint, device
    )

    if "epoch" in checkpoint and "validation_loss" in checkpoint:
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"(validation loss {checkpoint['validation_loss']:.4f})"
        )
    generated = generate_text(
        model,                         # the restored best model
        prompt=args.prompt,            # the starting text
        length=args.length,            # the number of new characters to generate
        char_to_index=char_to_index,   # converts prompt characters into IDs
        index_to_char=characters,      # converts predicted IDs back into characters
        device=device,                 # the computing device
        temperature=args.temperature,  # controls sampling randomness
        top_k=args.top_k,              # limits sampling to the top-scoring candidates
    )
    print(generated)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a character-level vanilla RNN"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a new RNN")
    train_parser.add_argument("--text-file", type=Path)
    train_parser.add_argument("--corpus-repeats", type=int, default=1)
    train_parser.add_argument("--sequence-length", type=int, default=40)
    train_parser.add_argument("--sequence-stride", type=int, default=10)
    train_parser.add_argument("--embedding-size", type=int, default=32)
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--num-layers", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--learning-rate", type=float, default=0.003)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "checkpoints" / "char_rnn.pt"
    )
    train_parser.add_argument(
        "--plot-output",
        type=Path,
        default=PROJECT_DIR / "examples" / "learning_curves.png",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(function=train)

    generate_parser = subparsers.add_parser(
        "generate", help="load a checkpoint and generate text"
    )
    generate_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "char_rnn.pt",
    )
    generate_parser.add_argument("--prompt", default="the ")
    generate_parser.add_argument("--length", type=int, default=300)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-k", type=int, default=5)
    generate_parser.add_argument("--seed", type=int, default=7)
    generate_parser.add_argument("--device", default="auto")
    generate_parser.set_defaults(function=predict)
    return parser


def validate_training_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command != "train":
        return
    positive_arguments = {
        "--corpus-repeats": args.corpus_repeats,
        "--sequence-length": args.sequence_length,
        "--sequence-stride": args.sequence_stride,
        "--embedding-size": args.embedding_size,
        "--hidden-size": args.hidden_size,
        "--num-layers": args.num_layers,
        "--batch-size": args.batch_size,
        "--epochs": args.epochs,
        "--learning-rate": args.learning_rate,
    }
    for name, value in positive_arguments.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    validate_training_arguments(args, parser)
    args.function(args)


if __name__ == "__main__":
    main()
