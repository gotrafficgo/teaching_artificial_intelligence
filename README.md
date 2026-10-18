# AI Learning Projects

Four independent, runnable projects introduce model-based simulation, image
classification, sequence modeling, and reinforcement learning. Each project includes a focused entry
point and a guide that explains the main concepts and experiments.

| Project | Learning focus | Entry point |
| --- | --- | --- |
| [Car-following simulation](car-following-simulation/README.md) | IDM, traffic flow, bottlenecks, and space-time diagrams | `car-following-simulation/main.py` |
| [Convolutional neural network](convolutional-neural-network/README.md) | MNIST, convolution, pooling, training, and prediction | `convolutional-neural-network/train.py` |
| [Recurrent neural network](recurrent-neural-network/README.md) | Character prediction, hidden states, BPTT, and text generation | `recurrent-neural-network/rnn_text_generation.py` |
| [Reinforcement learning](reinforcement-learning/README.md) | Maze navigation, rewards, exploration, and tabular Q-learning | `reinforcement-learning/q_learning.py` |

## Recommended learning path

1. Start with the car-following simulation to distinguish a hand-designed
   behavioral model from its simulation environment.
2. Use the CNN to see how model parameters are learned from labeled images.
3. Use the RNN to extend the same training ideas to ordered sequences.
4. Use the maze example to learn Q-learning through interaction and rewards.

The projects are independent. The CNN and RNN share PyTorch, but they do not
exchange data or model outputs.

## Set up the environment

Python 3.11 or newer is recommended. Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Verify the installation:

```bash
python -c "import torch, torchvision; print(torch.__version__)"
```

## Run the projects

### Car-following simulation

Run an experiment and save its time-space diagram without opening a window:

```bash
python car-following-simulation/main.py --experiment 1 \
  --output car-following-simulation/results/experiment-1.png --no-show
```

### Convolutional neural network

Generate a sample digit, then classify it with the included checkpoint:

```bash
python convolutional-neural-network/generate_test_digit.py
python convolutional-neural-network/predict.py \
  convolutional-neural-network/examples/test_digit_7.png
```

Train a new model with a short initial run:

```bash
python convolutional-neural-network/train.py --epochs 3
```

The first training run downloads MNIST.

### Recurrent neural network

Train the character-level RNN, select the best checkpoint, and generate text:

```bash
python recurrent-neural-network/run_rnn_example.py
```

## Run the available tests

```bash
cd convolutional-neural-network
python -m unittest -v test_cnn.py
```

See each project guide for model details, tensor shapes, and suggested
experiments. Deactivate the environment with `deactivate` when finished.
