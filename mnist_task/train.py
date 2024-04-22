import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance  # For Euclidean distance calculation
from tqdm import tqdm

class Classifier:
    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.num_classes = 10  # Number of classes in MNIST

    def load_data(self, data, test_sample_size=100, train_sample_size=1000):
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = data
        self.train_data = self.train_data.reshape(-1, 28*28) / 255.0  # Flatten and normalize
        self.test_data = self.test_data.reshape(-1, 28*28) / 255.0  # Flatten and normalize
        
        self.test_labels , self.test_data  = self.test_labels[:test_sample_size]  , self.test_data[:test_sample_size]
        self.train_labels, self.train_data = self.train_labels[:train_sample_size], self.train_data[:train_sample_size]

    def predict(self, data):

        predictions = np.empty(data.shape[0], dtype=self.train_labels.dtype)
        for i in tqdm(range(data.shape[0]), desc="Predicting"):
            distances = distance.cdist([data[i]], self.train_data, 'euclidean')
            nearest_index = np.argmin(distances)
            predictions[i] = self.train_labels[nearest_index]
        return predictions

    def get_confusion_matrix(self, predictions, train_or_test="test"):
        """
        Computes and plots the confusion matrix for specified data.

        Args:
            train_or_test (str): 'train' to use training data, 'test' to use testing data.
        """
        if train_or_test == "train":
            labels = self.train_labels
        else:
            labels = self.test_labels

        
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

        for true, pred in zip(labels, predictions):
            confusion_matrix[true][pred] += 1

        return confusion_matrix

    def get_error_rate(self, predictions, ground_truth):
        return np.mean(predictions != ground_truth) * 100
    
