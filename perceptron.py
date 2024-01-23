import numpy as np
import matplotlib.pyplot as plt

# The class for representing the perceptron
class Perceptron:
    def __init__(self, W: np.ndarray, b: float):
        # Initialize the weights and the bias
        self.W = W
        self.b = b

    def activate(self, inputs: np.ndarray) -> int:
        # Calculate the weighted sum and add bias
        weighted_sum = np.dot(inputs, self.W) + self.b
        # Get and return the output of the activation function
        return unit_step_actiavtion(weighted_sum)

    def update_w_b(self, W: np.ndarray, b: float):
        # Update the weights and the bias
        self.W = W
        self.b = b

    def get_w_b(self):
        return self.W, self.b

def unit_step_actiavtion(input: float) -> int:
    output = 1 if input >= 0 else 0
    return output

def decision_boundary(W: np.ndarray, b: float, x1: float):
    return (-W[0] * x1 - b) / W[1]

def plot_decision_boundary(X: np.ndarray, Y: np.ndarray, epoch: int, W: np.ndarray, b: float):
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, marker='o')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'Epoch {epoch + 1}')
    plt.axline((0, decision_boundary(W, b, 0)), slope=W[0] / -W[1], color='black', linestyle='--', label='Decision Boundary')
    plt.legend()
    # Add axes lines
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-3, 3)  # Set x-axis limits
    plt.ylim(-3, 3)  # Set y-axis limits
    plt.show()

def train_perceptron(train_data: np.ndarray, target_data: np.ndarray, perceptron: Perceptron, lr: float = 1, max_epochs: int = 100):
    # Training the perceptron
    for epoch in range(max_epochs):
        errors_sum: float = 0
        for X, Y in zip(train_data, target_data):
            # Calculate the perceptron output
            output = perceptron.activate(X)

            # Calculate the error
            e = Y - output
            errors_sum += abs(e)

            # Get the weights and bias of the the current state of perceptron
            W, b = perceptron.get_w_b()

            # Calculate new weights and bias
            W += lr * e * X
            b += lr * e

            # Update the weights and bias of the perceptron
            perceptron.update_w_b(W, b)

        if errors_sum == 0:
            # It should say that the training was done in the prev step
            print(f'Perceptron fully trained in epoch {epoch}.')
            return

        # Print decision boundary in the form X2 = f(X1)
        print(f'Decision Boundary (Epoch {epoch + 1}): X2 = {-W[0] / W[1]} * X1 + {-b / W[1]}\n\n')
        plot_decision_boundary(train_data, target_data, epoch, W, b)

def main():
    # Training data for AND gate
    X: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y: np.ndarray = np.array([0, 0, 0, 1])

    # Initial weights and bias
    W: np.ndarray = np.array([0.5, 1])
    b: float = -1

    # Parameters
    learning_rate: float = 1
    epochs: int = 100

    perceptron: Perceptron = Perceptron(W, b)

    train_perceptron(X, Y, perceptron, lr = learning_rate, max_epochs = epochs)