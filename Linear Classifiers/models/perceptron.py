"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.decay = 0.1

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = x_train.shape
        self.w = np.random.uniform(low=-1.0, high=1.0, size=(self.n_class,D)) * 0.01
        for epoch in range(self.epochs):
            for i in range(N):
                for c in range(self.n_class):
                    if self.w[c].T @ x_train[i] > self.w[y_train[i]].T @ x_train[i]:
                        self.w[y_train[i]] += self.lr * x_train[i]
                        self.w[c] -= self.lr * x_train[i]
            self.lr *= self.decay
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N, D = X_test.shape
        pred_y = np.zeros(N)
        for i in range(N):
            pred_y[i] = np.argmax(self.w @ X_test[i])
        return pred_y

    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100