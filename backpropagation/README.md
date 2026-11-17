# Backpropagation Neural Network from Scratch

A hands-on implementation of a feedforward neural network and the backpropagation algorithm using **Python and NumPy**, built to understand how neural networks learn through gradient descent.

This project explores the mathematical foundations of backpropagation, implements forward propagation and gradient calculations from scratch, and investigates how different learning rates affect neural network training.

The implementation uses the XOR problem as a learning experiment.

## 1. Learning Objectives

The main objectives of this project were:

- Understand how artificial neural networks process input data.
- Implement forward propagation using matrix multiplication.
- Understand activation functions and their derivatives.
- Derive and implement backpropagation.
- Calculate gradients for weights and biases.
- Update network parameters using gradient descent.
- Understand how learning rates influence training.
- Visualize training loss and evaluate model convergence.

## 2. Neural Network Architecture

We implemented a fully connected feedforward neural network with the following architecture:

**2 → 3 → 1**

| Layer | Neurons | Description |
|---|---:|---|
| Input | 2 | Receives two binary inputs |
| Hidden | 3 | Learns intermediate representations |
| Output | 1 | Produces the binary prediction |

The network uses the sigmoid activation function in both the hidden and output layers.

### Architecture

```text
       Input Layer       Hidden Layer       Output Layer

          X1  ───────────► H1
            \              │
             \             │
              ───────────► H2 ───────────► Y
             /             │
            /              │
          X2  ───────────► H3
```

### Parameter dimensions

| Parameter | Shape | Description |
|---|---|---|
| W1 | (3, 2) | Input-to-hidden weights |
| b1 | (3, 1) | Hidden layer biases |
| W2 | (1, 3) | Hidden-to-output weights |
| b2 | (1, 1) | Output layer bias |

The general weight matrix rule is:

\[
W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}
\]

where \(n_l\) represents the number of neurons in the current layer.

## 3. Mathematical Implementation

### 3.1 Activation Function

We implemented the sigmoid activation function:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Its derivative is:

\[
\sigma'(z)=\sigma(z)(1-\sigma(z))
\]

Sigmoid maps values into the range (0, 1), making it suitable for this binary-output experiment.

### 3.2 Forward Propagation

Forward propagation calculates the network's prediction.

**Hidden layer:**

\[
Z_1=W_1X+b_1
\]

\[
A_1=\sigma(Z_1)
\]

**Output layer:**

\[
Z_2=W_2A_1+b_2
\]

\[
A_2=\sigma(Z_2)
\]

Here, \(A_2\) is the predicted output.

### 3.3 Loss Function

We used the half squared-error loss:

\[
L=\frac{1}{2}\sum(A_2-Y)^2
\]

The factor of 1/2 simplifies differentiation.

For each training example, the loss measures the difference between the predicted output and the actual target.

### 3.4 Backpropagation

Backpropagation calculates how much each weight and bias contributes to the prediction error.

**Output layer error:**

\[
\delta_2=(A_2-Y)\odot A_2(1-A_2)
\]

**Output layer gradients:**

\[
dW_2=\delta_2 A_1^T
\]

\[
db_2=\delta_2
\]

**Hidden layer error:**

\[
\delta_1=(W_2^T\delta_2)\odot A_1(1-A_1)
\]

**Hidden layer gradients:**

\[
dW_1=\delta_1 X^T
\]

\[
db_1=\delta_1
\]

The gradients are propagated backward from the output layer to the hidden layer using the chain rule.

### 3.5 Gradient Descent

After calculating gradients, we update the parameters:

\[
W_{\text{new}}=W_{\text{old}}-\eta dW
\]

\[
b_{\text{new}}=b_{\text{old}}-\eta db
\]

Here, \(\eta\) is the learning rate.

The learning rate controls the size of each parameter update.

## 4. Dataset: XOR Problem

We used the XOR (exclusive OR) logical operation as our training dataset.

| Input 1 | Input 2 | Expected Output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR is a useful experiment because it is not linearly separable.

A single-layer perceptron cannot correctly learn XOR using a linear decision boundary.

A neural network with a hidden layer can learn the required nonlinear decision boundary.

## 5. Implementation Process

The project was implemented incrementally to understand each component rather than relying on an existing deep learning framework.

1. **Activation function:** Implemented sigmoid and its mathematical derivative.
2. **Network architecture:** Created a 2–3–1 feedforward neural network.
3. **Forward propagation:** Implemented matrix multiplication, bias addition, and activation.
4. **Loss calculation:** Implemented half squared-error loss.
5. **Backpropagation:** Derived and implemented gradients using the chain rule.
6. **Parameter updates:** Implemented gradient descent manually.
7. **Single-example experiment:** Verified forward propagation, gradients, and one parameter update.
8. **XOR training:** Trained the network using stochastic gradient descent.
9. **Loss visualization:** Used Matplotlib to visualize the training process.
10. **Learning-rate experiments:** Compared four different learning rates while keeping the other experimental settings unchanged.

## 6. Learning-Rate Experiments

We conducted four experiments using different learning rates.

The following settings were kept constant:

| Setting | Value |
|---|---|
| Network architecture | 2–3–1 |
| Dataset | XOR |
| Epochs | 10,000 |
| Initial weights | Same for every experiment |
| Initial biases | Same for every experiment |
| Training order | Same |
| Optimization | Stochastic Gradient Descent |
| Activation | Sigmoid |
| Loss | Half squared error |

Only the learning rate was changed.

### Experiment Results

| Learning Rate | Loss at 1,000 Epochs | Loss at 5,000 Epochs | Final Loss |
|---:|---:|---:|---:|
| 0.5 | 0.124979 | 0.000524 | 0.00017391 |
| 1.0 | 0.013384 | 0.000174 | 0.00007333 |
| 2.0 | 0.000731 | 0.000072 | 0.00003319 |
| 3.0 | 0.000368 | 0.000046 | **0.00002156** |

### Experiment 1: Learning Rate = 0.5

- Final loss: 0.00017391
- The network successfully learned the XOR operation.
- Training loss decreased steadily.
- This served as our baseline experiment.

### Experiment 2: Learning Rate = 1.0

- Final loss: 0.00007333
- Training converged considerably faster than the baseline.
- Loss at epoch 1,000 was already much lower than the baseline.
- The final loss was approximately 57.9% lower than the baseline.

### Experiment 3: Learning Rate = 2.0

- Final loss: 0.00003319
- Training converged successfully.
- The network achieved a lower loss than the previous experiments.
- No divergence or NaN values were observed.

### Experiment 4: Learning Rate = 3.0

- Final loss: 0.00002156
- This produced the lowest final loss among our four experiments.
- Training remained stable throughout the observed checkpoints.
- Increasing the learning rate from 2.0 to 3.0 improved the final result in this particular setup.

### What We Learned

1. A larger learning rate can help a network converge faster.
2. A learning rate that is too large can cause unstable updates or divergence.
3. The best learning rate depends on the network, initialization, dataset, and optimization process.
4. A lower final training loss does not necessarily guarantee better generalization.
5. Learning-rate experiments should change one variable at a time to make comparisons meaningful.

**Important:** Learning rate 3.0 was the best among the four tested values for this particular experiment. It is not a universally optimal learning rate.

## 7. Final XOR Predictions

The following predictions were obtained using learning rate 3.0.

| Input | Expected | Predicted |
|---|---:|---:|
| [0, 0] | 0 | 0.006001 |
| [0, 1] | 1 | 0.993786 |
| [1, 0] | 1 | 0.993742 |
| [1, 1] | 0 | 0.007659 |

The network successfully learned all four XOR input-output combinations.

## 8. Loss Visualization

We used Matplotlib to visualize training loss against the number of epochs.

The loss curve helps us understand:

- Whether the network is learning.
- How quickly the loss decreases.
- Whether training is converging.
- Whether optimization becomes unstable.

Loss curves are saved in the `results/` directory.

## 9. Project Structure

```text
backpropagation/
│
├── activations.py
├── network.py
├── losses.py
├── main.py
├── results/
│   ├── loss_curve.png
│   └── loss_curve_lr_1.0.png
│
├── pyproject.toml
├── uv.lock
└── README.md
```

The loss curve filenames shown above are examples; keep the actual filenames generated by your experiments.

## 10. Technologies Used

- Python
- NumPy
- Matplotlib
- uv for Python environment and dependency management

No PyTorch or TensorFlow was used to implement the network. Forward propagation, backpropagation, and gradient descent were implemented manually using NumPy.

## 11. How to Run

Clone the repository or navigate to the project directory.

Install dependencies:

```bash
uv sync
```

Run the implementation:

```bash
uv run main.py
```

The program trains the XOR neural network, prints training loss and final predictions, and saves the loss visualization.

## 12. Limitations and Future Work

This is an educational implementation focused on understanding the mechanics of backpropagation. It is inspired by the foundational ideas in:

**Rumelhart, Hinton, and Williams (1986) — Learning representations by back-propagating errors.**

It is not an exact reproduction of the paper's original experiments.

Potential improvements:

- Implement numerical gradient checking using finite differences.
- Compare analytical gradients against numerical gradients.
- Experiment with different weight initialization strategies.
- Compare sigmoid, tanh, and ReLU activation functions.
- Compare stochastic gradient descent with batch and mini-batch gradient descent.
- Experiment with more network architectures.
- Test learning-rate stability across multiple random initializations.
- Implement momentum-based gradient descent.

## 13. Key Takeaway

This project helped me understand how a neural network learns internally rather than treating it as a black-box model.

I implemented forward propagation, derived backpropagation using the chain rule, calculated gradients, and trained a neural network using gradient descent.

The XOR experiment demonstrated why hidden layers and nonlinear activation functions are important, while the learning-rate experiments showed how optimization hyperparameters influence convergence.

The goal was not simply to make the network predict XOR correctly, but to understand the mathematical process that allows a neural network to learn.