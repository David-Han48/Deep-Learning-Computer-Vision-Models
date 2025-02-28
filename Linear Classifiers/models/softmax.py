"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, decay: float, seed: int, temp):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.decay = decay
        self.seed = seed
        self.temp = temp

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        N, D = X_train.shape
        max_scores = np.max(X_train @ self.w.T, axis=1, keepdims=True)
        P = np.exp((X_train @ self.w.T - max_scores)/self.temp)
        P /= np.sum(P, axis=1, keepdims=True)
        g = np.zeros(self.w.shape)
        for i in range(N):
            for c in range(self.n_class):
                if c == y_train[i]:
                    g[c] += (P[i,c] - 1) * X_train[i]
                else:
                    g[c] += P[i,c] * X_train[i]
        g += self.reg_const / N * self.w
        return g

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.
        
        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = X_train.shape
        np.random.seed(self.seed)
        self.w = np.random.uniform(low=-1.0, high=1.0, size=(self.n_class,D)) * 0.01
        # state = np.random.get_state()
        # print(f'seed: {state[1][0]}')
        batch_size = 128
        
        for epoch in range(self.epochs):
            permutation = np.random.permutation(N)
            X_train = X_train[permutation]
            y_train = y_train[permutation]
            for i in range(0, N, batch_size):
                g = self.calc_gradient(X_train[i:i+batch_size], y_train[i:i+batch_size])
                self.w -= self.lr * g
            self.lr *= self.decay
        return

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