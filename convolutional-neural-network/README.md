# Handwritten Digit Recognition with a CNN

In this project, you will train a convolutional neural network to recognize
handwritten digits and use the trained model to classify a new image. The
complete PyTorch workflow is:

```text
Load data -> Train -> Validate -> Save -> Predict
```

The CNN is intentionally small so that its architecture, tensor shapes, training
loop, evaluation, and inference code are easy to inspect.

## Project files

| File | Purpose |
| --- | --- |
| `model.py` | Defines the CNN architecture and forward pass |
| `train.py` | Trains, validates, saves the best checkpoint, and plots learning curves |
| `image_utils.py` | Converts an external digit image into MNIST-like model input |
| `predict.py` | Loads a checkpoint and predicts one digit |
| `predict_step_by_step.py` | Runs the trained layers individually and prints tensor shapes |
| `generate_test_digit.py` | Generates a sample handwritten digit |
| `test_cnn.py` | Runs lightweight automated checks |

## 1. Set up the shared environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd convolutional-neural-network
```

Python 3.11 or newer is recommended. On Windows PowerShell, activate the
environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## 2. Inspect the model

```bash
python model.py
```

Tensor shapes:

```text
[B, 1, 28, 28]
  -> Conv/ReLU/Pool -> [B, 16, 14, 14]
  -> Conv/ReLU/Pool -> [B, 32, 7, 7]
  -> Flatten/Linear -> [B, 10] logits
```

Trainable parameters:

| Layer | Calculation | Parameters |
| --- | ---: | ---: |
| Conv1 | `16 x (1 x 3 x 3 + 1)` | 160 |
| Conv2 | `32 x (16 x 3 x 3 + 1)` | 4,640 |
| Linear | `10 x (1,568 + 1)` | 15,690 |
| **Total** |  | **20,490** |

The `+1` represents one bias for each output channel or unit.

## 3. Train and validate

```bash
python train.py
```

The first run downloads MNIST. The script uses a fixed random seed and splits
the dataset into:

```text
55,000 training images
 5,000 validation images
```

- The training set is used to update the parameters.
- The validation set is used to select the best checkpoint.

The default training configuration is:

```text
epochs        = 20
batch size    = 128
learning rate = 0.001
optimizer     = Adam
loss          = Cross-Entropy Loss
```

The best checkpoint is saved to:

```text
checkpoints/mnist_cnn.pt
```

Training and validation curves are saved to:

```text
examples/learning_curves.png
```

The central training step is:

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

`CrossEntropyLoss` expects raw logits and applies the required log-Softmax
operation internally. Do not apply Softmax before calculating the training loss.

Run `python train.py --help` to view the available options.

## 4. Make a prediction

Generate the included sample digit and run inference:

```bash
python generate_test_digit.py
python predict.py examples/test_digit_7.png \
  --save-preprocessed examples/preprocessed.png
```

The script prints the predicted digit and all ten Softmax probabilities. The
saved preview shows the image geometry before tensor standardization.

You can also use your own PNG or JPEG:

```bash
python predict.py path/to/my_digit.png
```

The preprocessing function converts the image to grayscale, detects whether it
needs to be inverted, crops the foreground, resizes it, centers it, and applies
the MNIST normalization statistics. This is a simple preprocessing heuristic;
shadows, transparent backgrounds, border-touching digits, or multiple digits
may produce poor results.

## 5. Inspect one prediction

```bash
python predict_step_by_step.py examples/test_digit_7.png
```

This runs the real trained layers one at a time:

```text
 1. Normalized input                 [1, 1, 28, 28]
 2. Convolution 1                    [1, 16, 28, 28]
 3. ReLU 1                           [1, 16, 28, 28]
 4. Max pooling 1                    [1, 16, 14, 14]
 5. Convolution 2                    [1, 32, 14, 14]
 6. ReLU 2                           [1, 32, 14, 14]
 7. Max pooling 2                    [1, 32, 7, 7]
 8. Flatten                          [1, 1,568]
 9. Linear layer                     [1, 10] logits
10. Softmax                          [1, 10] probabilities
11. Prediction                       one class from 0 to 9
```

A Softmax probability is not necessarily a calibrated confidence score. A model
can assign a high probability to an incorrect prediction, particularly when the
input differs from the MNIST training distribution.

## 6. Run the tests

```bash
python -m unittest -v test_cnn.py
```

The tests verify the model output shape, Softmax probabilities, image
preprocessing, blank-image error handling, and checkpoint loading.
