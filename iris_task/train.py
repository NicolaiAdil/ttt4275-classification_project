import numpy as np
from sklearn.preprocessing import OneHotEncoder


def MSE(predictions, ground_truth):
    error = predictions - ground_truth
    return 1/2 * np.sum(np.matmul(error, error.transpose()))

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def get_gradient_MSE(predictions, ground_truth, data):
        grad_g_MSE = predictions - ground_truth
        grad_z_g = np.multiply(predictions, (1 - predictions))
        grad_W_z = data.transpose()

        grad_W_MSE = np.multiply(grad_g_MSE, grad_z_g).transpose().dot(grad_W_z)
        return grad_W_MSE

class LinearClassifier:
    def __init__(self, alpha=0.0025, max_iter=1000, tol=1e-3, num_classes=3):
        self.alpha = alpha  # Step factor
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence
        self.weights = None  # Weights for linear classifier
        self.num_classes = num_classes  # Number of classes

    def forward_pass(self, data):
        """
        Computes the forward pass of the model.

        Args:
            data (numpy.ndarray): The input data of shape (num_features, num_samples).

        Returns:
            numpy.ndarray: The predictions of the model.
        """
        return sigmoid(np.matmul(self.weights, data)).transpose()
    
    def plot_confusion_matrix(self, data, ground_truth):
        predictions = self.predict(data)
        confusion_matrix = np.zeros([3, 3], dtype=int)
        for i in range(len(predictions)):
            confusion_matrix[np.argmax(predictions[i])][ground_truth[i]] += 1
        return confusion_matrix

    def train(self, data: np.ndarray, ground_truth: np.ndarray, test_data: np.ndarray,
             test_ground_truth: np.ndarray, verbose: bool, num_classes: int = 3) -> list:
        """
        Trains the model using the given data and ground truth labels.

        Args:
            data (numpy.ndarray): The input data of shape (num_samples, num_features).
            ground_truth (numpy.ndarray): The ground truth labels of shape (num_samples,).
            num_classes (int, optional): The number of classes. Defaults to 3.

        Returns:
            list: A list containing the loss for each iteration.
        """
        self.num_classes = num_classes
        num_samples, num_features = data.shape
        self.weights = np.zeros([num_classes, num_features+1], dtype=float)  # Initialize weights
        if verbose:
            print(f"Initial weights shape: {self.weights.shape}\n-----------------")

        data = np.concatenate((data, np.ones([data.shape[0], 1], dtype=int)), axis=1).transpose()  # Add bias term
        test_data = np.concatenate((test_data, np.ones([test_data.shape[0], 1], dtype=int)), axis=1).transpose()  # Add bias term
        if verbose:
            print(f"Data shape: {data.shape}\n-----------------")
        # Store the loss for each iteration
        loss_vector = [float('inf')]

        encoder = OneHotEncoder()
        ground_truth = encoder.fit_transform(ground_truth.reshape(-1, 1)).toarray()
        test_ground_truth = encoder.fit_transform(test_ground_truth.reshape(-1, 1)).toarray()

        for i in range(self.max_iter):
            prev_loss = loss_vector[i]

            # Compute forward pass
            predictions = self.forward_pass(data)  # Predictions

            # Update weights based on gradient descent method
            gradient = get_gradient_MSE(predictions, ground_truth, data)  # Derivative of loss function (MSE)
            self.weights -= self.alpha * gradient

            # Test the current model on the test data
            test_predictions = self.forward_pass(test_data)
            test_loss = MSE(test_predictions, test_ground_truth)

            # Check for convergence
            if abs(test_loss - prev_loss) < self.tol:
                break

            loss_vector.append(test_loss)
        return loss_vector

    def predict(self, data):
        return np.dot(data, self.weights)