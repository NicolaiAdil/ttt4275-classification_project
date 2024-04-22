import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance  # For Euclidean distance calculation

class Classifier:
    def __init__(self):
        self.train_data = None
        self.train_labels = None

    def load_data(self, data):
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = data
        self.train_data = self.train_data.reshape(-1, 28*28) / 255.0  # Flatten and normalize
        self.test_data = self.test_data.reshape(-1, 28*28) / 255.0  # Flatten and normalize

    def predict(self, test_data, num_neighbors=1):
        # Find Euclidean distances from test data to all training data
        distances = distance.cdist(test_data, self.train_data, 'euclidean')
        # Get the indices of the smallest distances
        nearest_indices = np.argmin(distances, axis=1)
        # Return the most common class among the nearest
        return self.train_labels[nearest_indices]

    def get_confusion_matrix(self, predictions, ground_truth, num_classes=10):
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true_class, predicted_class in zip(ground_truth, predictions):
            confusion_matrix[true_class][predicted_class] += 1
        return confusion_matrix

    def get_error_rate(self, predictions, ground_truth):
        return np.mean(predictions != ground_truth)
    
    