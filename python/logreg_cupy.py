from sklearn.model_selection import train_test_split
from cupyx.profiler import benchmark
from sklearn import datasets
import numpy as np
import cupy as cp

NUM_REPS = 20
NUM_EPOCHS = [150, 300, 450, 600, 750, 900, 1050, 1200]


def get_dataset():
    iris = datasets.load_iris()
    X = iris.data
    Y = np.where(iris.target == 0, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return cp.array(X_train), cp.array(X_test), cp.array(y_train), cp.array(y_test)


def sigmoid(X):
    return 1 / (1 + cp.exp(-X))


def logistic_regression_loss(W, X, y):
    N, D = X.shape
    y_hat = sigmoid(X @ W)
    y_hat = cp.clip(y_hat, 0.0001, 0.9999)

    loss = (y * cp.log(y_hat) + (1 - y) * cp.log(1 - y_hat)) / -N
    grads = (X.T @ (y_hat - y)) / N

    return cp.sum(loss), grads


def train(X, y, learning_rate, num_epochs):
    # X has shape (num_samples, 4), y has shape (num_samples,)
    N, D = X.shape
    W = 1e-4 * cp.random.randn(D)

    for i in range(num_epochs):
        loss, grads = logistic_regression_loss(W, X, y)
        W -= learning_rate * grads


def main():
    X_train, X_test, y_train, y_test = get_dataset()

    for num_epochs in NUM_EPOCHS:
        avg_time = benchmark(
            train,
            (
                X_train,
                y_train,
                1,
                num_epochs,
            ),
            n_repeat=NUM_REPS,
        )

        print(
            f"CuPy takes {avg_time.gpu_times.mean() * 1000} ms on average to run {num_epochs} epochs of logistic regression on iris setsosa"
        )


if __name__ == "__main__":
    main()
