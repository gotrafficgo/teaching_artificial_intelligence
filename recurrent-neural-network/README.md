# Character-Level Text Generation with an RNN

In this project, you will train a recurrent neural network to predict the next
character and then use the trained model to generate new text. The complete
workflow is implemented in one program:

```text
Text -> Character IDs -> Train/Validate -> Learning Curve -> Best Checkpoint -> Generate
```

The program deliberately uses a vanilla `nn.RNN`, rather than an LSTM or GRU.
This keeps the code close to the standard hidden-state equation:

```text
h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)
```

## Project files

| File | Purpose |
| --- | --- |
| `rnn_text_generation.py` | Defines the RNN, prepares sequences, trains the model, selects a checkpoint, and generates text |
| `run_rnn_example.py` | Runs training, model selection, and text generation |
| `data/alice_in_wonderland.txt` | Provides the default public-domain training corpus |
| `data/mini_english_corpus.txt` | Provides a smaller original corpus for quick experiments |
| `data/SOURCE.txt` | Records the source and preparation of the default corpus |
| `requirements.txt` | Lists the PyTorch and plotting dependencies |

## 1. Set up the shared environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd recurrent-neural-network
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## 2. Train the model

The program includes the public-domain text of *Alice's Adventures in
Wonderland*, so no dataset download is needed. The text is stored in
`data/alice_in_wonderland.txt`, where you can inspect or replace it without
editing the Python program. The source is documented in `data/SOURCE.txt`.

Run the complete example:

```bash
python run_rnn_example.py
```

The command trains for 20 epochs, plots the learning curve, saves the checkpoint
with the lowest validation loss, reloads that checkpoint, and generates text.
To control training and generation separately, use the commands below.

Train the model:

```bash
python rnn_text_generation.py train
```

This command trains and evaluates the model, saves the best checkpoint, and
plots the learning curves. Text generation is handled separately by the
`generate` command below.

The non-repeated corpus is shuffled with a fixed random seed and split at
complete paragraph boundaries. No paragraph appears in both the training and
validation sets. During each epoch, the program prints both cross-entropy
losses:

```text
Epoch 01/20 | training loss 2.9814 | training accuracy 18.42% |
validation loss 2.7441 | validation accuracy 22.17%
```

The optimizer updates the model using only the training sequences. Validation
sequences measure how well the current model handles data that was not used for
parameter updates.

The checkpoint with the lowest validation loss is saved to:

```text
checkpoints/char_rnn.pt
```

The learning curve is saved to:

```text
examples/learning_curves.png
```

The figure follows the same visual design as the CNN learning curves. The left
panel shows cross-entropy loss, and the right panel shows next-character
accuracy. Use both panels to diagnose training:

- If both losses remain high, the model is underfitting.
- If both losses decrease, the model is learning useful sequence patterns.
- If training loss decreases while validation loss increases, the model is
  overfitting.
- The vertical dashed line marks the checkpoint selected for generation.

The RNN checkpoint is selected by minimum validation loss, not maximum
next-character accuracy. Loss measures the quality of the complete predicted
probability distribution, which is important when the model samples text.

## 3. Generate text from the saved model

```bash
python rnn_text_generation.py generate \
  --prompt "the moon moves" \
  --length 300
```

Generation is autoregressive: the predicted character becomes the next input,
and the hidden state is passed to the following time step.

## 4. Train with another English text file

```bash
python rnn_text_generation.py train \
  --text-file my_corpus.txt \
  --sequence-length 80 \
  --epochs 30 \
  --output checkpoints/my_rnn.pt

python rnn_text_generation.py generate \
  --checkpoint checkpoints/my_rnn.pt \
  --prompt "once upon" \
  --length 500
```

The generation prompt may only contain characters that occurred in the
training corpus.

## 5. Trace the data through the model

Each training example consists of two shifted sequences:

```text
input:   t h e   m o o
target:  h e   m o o n
```

For a batch with shape `[B, T]`, the important tensor shapes are:

```text
token IDs                 [B, T]
embedding                 [B, T, embedding_size]
RNN output                [B, T, hidden_size]
logits                    [B, T, vocabulary_size]
```

`CrossEntropyLoss` compares the logits at every time step with the next true
character. Calling `loss.backward()` performs backpropagation through the
unrolled time steps. Gradient clipping is included because recurrent networks
can suffer from exploding gradients.

The model is not selected from the final epoch automatically. After every
epoch, the program compares the new validation loss with the best value seen so
far. It saves the model only when validation loss improves.

After the basic program works, change one setting at a time and compare the
results:

- Try temperatures `0.3`, `0.8`, and `1.2`.
- Set `--top-k 0` to sample from the complete vocabulary.
- Increase `--sequence-length` and observe how sequence length affects learning.
- Replace `nn.RNN` with `nn.GRU` or `nn.LSTM`, then compare the generated text.

Run `python rnn_text_generation.py train --help` or
`python rnn_text_generation.py generate --help` to see every option.
