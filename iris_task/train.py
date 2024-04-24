import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def MSE(predictions, ground_truth):
    error = predictions - ground_truth
    return 1 / 2 * np.sum(np.matmul(error, error.transpose()))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_gradient_MSE(predictions, ground_truth, data):
    grad_g_MSE = predictions - ground_truth
    grad_z_g = np.multiply(predictions, (1 - predictions))
    grad_W_z = data.transpose()

    grad_W_MSE = np.multiply(grad_g_MSE, grad_z_g).transpose().dot(grad_W_z)
    return grad_W_MSE


def train_and_plot_MSE(
    train_data, train_labels, test_data, test_labels, alphas, title, verbose
):
    for alpha in alphas:
        classifier = LinearClassifier(alpha=alpha)
        loss_vector, _, _ = classifier.train(
            train_data, train_labels, test_data, test_labels, verbose
        )
        plt.plot(loss_vector, label=f"Alpha: {alpha}")
    # Plot the MSE for each step length
    plt.title(f"MSE of the training data: {title}")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


class LinearClassifier:
    def __init__(self, alpha=0.0025, max_iter=1000, tol=1e-3):
        self.alpha = alpha  # Step factor
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence
        self.weights = None  # Weights for linear classifier

        self.data = None  # Input data
        self.ground_truth = None  # Ground truth labels
        self.test_data = None  # Test data
        self.test_ground_truth = None  # Test ground truth labels

    def forward_pass(self, data):
        """
        Computes the forward pass of the model.

        Args:
            data (numpy.ndarray): The input data of shape (num_features, num_samples).

        Returns:
            numpy.ndarray: The predictions of the model.
        """
        return sigmoid(np.matmul(self.weights, data)).transpose()

    def get_error_rate(self, predictions, ground_truth):
        """
        Computes the error rate of the model.

        Args:
            predictions (numpy.ndarray): The predictions of the model.
            ground_truth (numpy.ndarray): The ground truth labels.

        Returns:
            float: The error rate of the model.
        """
        error_rate = 0
        for i in range(len(predictions)):
            if np.argmax(predictions[i]) != np.argmax(ground_truth[i]):
                error_rate += 1
        return error_rate / len(predictions)

    def get_confusion_matrix(self, train_or_test: str = "train"):

        if train_or_test == "train":
            data = self.data
            ground_truth = self.ground_truth
        elif train_or_test == "test":
            data = self.test_data
            ground_truth = self.test_ground_truth
        else:
            raise ValueError("train_or_test must be either 'train' or 'test'.")

        predictions = self.forward_pass(data)
        confusion_matrix = np.zeros([3, 3], dtype=int)
        for i in range(len(predictions)):
            confusion_matrix[np.argmax(predictions[i])][np.argmax(ground_truth[i])] += 1
        return confusion_matrix

    def train(
        self,
        data: np.ndarray,
        ground_truth: np.ndarray,
        test_data: np.ndarray,
        test_ground_truth: np.ndarray,
        verbose: bool,
        num_classes: int = 3,
    ) -> list:
        """
        Trains the model using the given data and ground truth labels.

        Args:
            data (numpy.ndarray): The input data of shape (num_samples, num_features).
            ground_truth (numpy.ndarray): The ground truth labels of shape (num_samples,).
            num_classes (int, optional): The number of classes. Defaults to 3.

        Returns:
            list: A list containing the loss for each iteration.
            list: A list containing the error rate for each iteration.
        """
        num_samples, num_features = data.shape
        self.weights = np.zeros(
            [num_classes, num_features + 1], dtype=float
        )  # Initialize weights

        # Add bias term to the data (Bias trick), and update member variables
        data = np.concatenate(
            (data, np.ones([data.shape[0], 1], dtype=int)), axis=1
        ).transpose()  # Add bias term
        self.data = data

        test_data = np.concatenate(
            (test_data, np.ones([test_data.shape[0], 1], dtype=int)), axis=1
        ).transpose()  # Add bias term
        self.test_data = test_data

        if verbose:
            print(f"Data shape: {data.shape}\n-----------------")

        # One-hot encode the ground truth labels
        encoder = OneHotEncoder()
        ground_truth = encoder.fit_transform(ground_truth.reshape(-1, 1)).toarray()
        self.ground_truth = ground_truth

        test_ground_truth = encoder.fit_transform(
            test_ground_truth.reshape(-1, 1)
        ).toarray()
        self.test_ground_truth = test_ground_truth

        # Store the loss and error rate for each iteration
        loss_vector = [float("inf")]
        error_rate_vector = [float("inf")]
        error_rate_test_vector = [float("inf")]

        for i in range(self.max_iter):
            prev_loss = loss_vector[i]

            # Compute forward pass
            predictions = self.forward_pass(data)  # Predictions

            # Update weights based on gradient descent method
            gradient = get_gradient_MSE(
                predictions, ground_truth, data
            )  # Derivative of loss function (MSE)
            self.weights -= self.alpha * gradient

            # Find loss of currents weights on test data
            test_predictions = self.forward_pass(test_data)
            test_loss = MSE(test_predictions, test_ground_truth)

            # Check for convergence
            if abs(test_loss - prev_loss) < self.tol:
                break

            loss_vector.append(test_loss)
            error_rate_vector.append(
                self.get_error_rate(test_predictions, test_ground_truth)
            )
            error_rate_test_vector.append(
                self.get_error_rate(predictions, ground_truth)
            )

        return loss_vector, error_rate_vector, error_rate_test_vector
