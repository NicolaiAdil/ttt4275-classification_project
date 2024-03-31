import numpy as np


class LinearClassifier:
    def __init__(self, alpha=0.01, max_iter=1000, tol=1e-3):
        self.alpha = alpha  # Step factor
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence
        self.weights = None  # Weights for linear classifier

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        prev_loss = float('inf')

        for _ in range(self.max_iter):
            # Compute predictions
            predictions = np.dot(X, self.weights)

            # Compute loss and gradient
            loss = np.mean((predictions - y) ** 2)
            gradient = 2 * np.dot(X.T, (predictions - y)) / n_samples

            # Update weights
            self.weights -= self.alpha * gradient

            # Check for convergence
            if abs(loss - prev_loss) < self.tol:
                break

            prev_loss = loss

    def predict(self, X):
        return np.dot(X, self.weights)