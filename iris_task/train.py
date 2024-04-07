import numpy as np
from sklearn.preprocessing import OneHotEncoder


# Define the linear classifier (you can use any variant of gradient descent)
class LinearClassifier:
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-3):
        self.alpha = alpha  # Step factor
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence
        self.weights = None  # Weights for linear classifier

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def get_gradient(self, predictions, ground_truth, data):
        grad_g_MSE = predictions - ground_truth
        grad_z_g = np.multiply(predictions, (1 - predictions)).transpose()
        grad_W_z = data.transpose()
        print("Shapes:", grad_g_MSE.shape, grad_z_g.shape, grad_W_z.shape)
        grad_W_MSE = np.sum(grad_g_MSE.dot(grad_z_g).dot(grad_W_z))
        return grad_W_MSE

    def train(self, data: np.ndarray, ground_truth: np.ndarray, verbose: bool, num_classes: int = 3) -> list:
        """
        Trains the model using the given data and ground truth labels.

        Args:
            data (numpy.ndarray): The input data of shape (num_samples, num_features).
            ground_truth (numpy.ndarray): The ground truth labels of shape (num_samples,).
            num_classes (int, optional): The number of classes. Defaults to 3.

        Returns:
            list: A list containing the loss for each iteration.
        """
        num_samples, num_features = data.shape
        self.weights = np.zeros([num_classes, num_features+1], dtype=int)  # Initialize weights
        if verbose:
            print(f"Initial weights shape: {self.weights.shape}\n-----------------")

        data = np.concatenate((data, np.ones([data.shape[0], 1], dtype=int)), axis=1).transpose()  # Add bias term
        if verbose:
            print(f"Data shape: {data.shape}\n-----------------")
        # Store the loss for each iteration
        loss_vector = [float('inf')]

        encoder = OneHotEncoder()
        ground_truth = encoder.fit_transform(ground_truth.reshape(-1, 1)).toarray()

        for i in range(self.max_iter):
            prev_loss = loss_vector[i]

            # Compute loss and gradient
            print(self.weights.shape, data.shape)
            predictions = self.sigmoid(np.matmul(self.weights, data)).transpose()
            
            if verbose:
                print(f"Predictions shape: {predictions.shape}\n-----------------")
                print(predictions.transpose())

            loss = 1/2 * np.sum( np.matmul((predictions - ground_truth), (predictions - ground_truth).transpose()) )
            gradient = self.get_gradient(predictions, ground_truth, data)  # Derivative of loss function (MSE)

            # Update weights
            self.weights -= self.alpha * gradient

            # Check for convergence
            if abs(loss - prev_loss) < self.tol:
                break

            loss_vector.append(loss)
            return loss_vector

    def predict(self, data):
        return np.dot(data, self.weights)