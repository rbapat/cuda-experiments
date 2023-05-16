from sklearn.model_selection import train_test_split
from sklearn import datasets
import torch.utils.benchmark as benchmark
import numpy as np
import torch

NUM_REPS = 20
NUM_EPOCHS = [150, 300, 450, 600, 750, 900, 1050, 1200]


def get_dataset(device):
    iris = datasets.load_iris()
    X = iris.data
    Y = np.where(iris.target == 0, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return (
        torch.from_numpy(X_train).to(device).to(torch.float32),
        torch.from_numpy(X_test).to(device).to(torch.float32),
        torch.from_numpy(y_train).to(device).to(torch.float32),
        torch.from_numpy(y_test).to(device).to(torch.float32),
    )


def sigmoid(X):
    return 1 / (1 + torch.exp(-X))


def logistic_regression_loss(W, X, y):
    N, D = X.shape
    y_hat = sigmoid(X @ W)
    y_hat = torch.clip(y_hat, 0.0001, 0.9999)

    loss = (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)) / -N
    grads = (X.T @ (y_hat - y)) / N

    return torch.sum(loss), grads


def train(X, y, learning_rate, num_epochs, device):
    # X has shape (num_samples, 4), y has shape (num_samples,)
    N, D = X.shape
    W = 1e-4 * torch.randn(D, device=device)

    for i in range(num_epochs):
        loss, grads = logistic_regression_loss(W, X, y)
        W -= learning_rate * grads


def main():
    device = torch.device("cuda")
    X_train, X_test, y_train, y_test = get_dataset(device)

    for num_epochs in NUM_EPOCHS:
        t0 = benchmark.Timer(
            stmt=f"train(X, y, 1, {num_epochs}, device)",
            setup="from __main__ import train",
            globals={"X": X_train, "y": y_train, "device": device},
        )

        print(
            f"PyTorch takes {t0.timeit(NUM_REPS).mean * 1e3} ms on average to run {num_epochs} epochs of logistic regression on iris setsosa"
        )


if __name__ == "__main__":
    main()
