"""
Part 1: learning slowdown with the quadratic cost.

Recreates the two graphs at the start of Chapter 3 of Nielsen's
"Neural Networks and Deep Learning". A single sigmoid neuron with one
input is trained to map the input 1 to the output 0.

    Run 1: starting weight 0.6, starting bias 0.9  (output starts at 0.82)
    Run 2: starting weight 2.0, starting bias 2.0  (output starts at 0.98)

Both use the quadratic cost C = (y - a)^2 / 2, learning rate 0.15,
and 300 epochs. Run 2 starts further from the right answer, yet it
learns much more slowly at first. That is the learning slowdown.

Usage:  python3 part1_learning_slowdown.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

X = 1.0        # the single training input
Y = 0.0        # the desired output
ETA = 0.15     # learning rate
EPOCHS = 300


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train(w, b, eta=ETA, epochs=EPOCHS):
    """Gradient descent on one neuron. Returns the cost, output, weight
    and bias recorded at every epoch (including epoch 0)."""
    costs, outputs, weights, biases = [], [], [], []
    for _ in range(epochs + 1):
        a = sigmoid(w * X + b)
        costs.append(0.5 * (Y - a) ** 2)
        outputs.append(a)
        weights.append(w)
        biases.append(b)
        # dC/dw = (a - y) * sigma'(z) * x   and   dC/db = (a - y) * sigma'(z)
        delta = (a - Y) * a * (1 - a)
        w -= eta * delta * X
        b -= eta * delta
    return costs, outputs, weights, biases


def main():
    runs = [
        ("Starting weight 0.6, bias 0.9", 0.6, 0.9),
        ("Starting weight 2.0, bias 2.0", 2.0, 2.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, w0, b0) in zip(axes, runs):
        costs, outputs, weights, biases = train(w0, b0)
        ax.plot(range(EPOCHS + 1), costs, color="#2a78d6", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, EPOCHS)
        ax.set_ylim(0, 0.5)
        ax.grid(alpha=0.3)
        print(f"{title}")
        print(f"  output: {outputs[0]:.2f} -> {outputs[-1]:.2f}")
        print(f"  cost:   {costs[0]:.4f} -> {costs[-1]:.4f}")
        print(f"  final weight {weights[-1]:.2f}, final bias {biases[-1]:.2f}")
    axes[0].set_ylabel("Cost")
    fig.suptitle("Learning slowdown: one sigmoid neuron, quadratic cost, "
                 "learning rate 0.15")
    fig.tight_layout()
    fig.savefig("part1_learning_slowdown.png", dpi=150)
    print("Saved part1_learning_slowdown.png")


if __name__ == "__main__":
    main()
