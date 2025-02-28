"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.
        if z >= 0:
            return 1/(1+np.exp(-z))
        else:
            return np.exp(z)/(1+np.exp(z))

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. 
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function 

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = x_train.shape
        self.w = np.random.uniform(low=-1.0, high=1.0, size=D) * 0.01

        for epoch in range(self.epochs):
            for i in range(N):
                self.w -= self.lr * x_train[i] * (self.sigmoid(self.w.T @ x_train[i]) - y_train[i])
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N, D = X_test.shape
        y_pred = np.zeros(N)
        for i in range(N):
            if self.w.T @ X_test[i] > self.threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred