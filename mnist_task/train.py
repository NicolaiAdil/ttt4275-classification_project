import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance  # For Euclidean distance calculation
from tqdm import tqdm


class Classifier:
    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.num_classes = 10  # Number of classes in MNIST

    def load_data(self, data, test_sample_size=1000, train_sample_size=10000):
        self.train_data, self.train_labels, self.test_data, self.test_labels = (
            data["trainv"],
            data["trainlab"],
            data["testv"],
            data["testlab"],
        )

        # The handout data was strangely structured, hack to fix.
        self.train_labels = np.array([label[0] for label in data["trainlab"]])
        self.test_labels = np.array([label[0] for label in data["testlab"]])

        # Flatten and normalize
        self.train_data = self.train_data.reshape(-1, 28 * 28) / 255.0
        self.test_data = self.test_data.reshape(-1, 28 * 28) / 255.0

        # Slice the data to reduce computation time
        self.test_labels, self.test_data = (
            self.test_labels[:test_sample_size],
            self.test_data[:test_sample_size],
        )
        self.train_labels, self.train_data = (
            self.train_labels[:train_sample_size],
            self.train_data[:train_sample_size],
        )

    def process_in_batches(
        self, test_data, train_data, train_labels, k=1, batch_size=1000
    ):
        """
        Processes the test data in batches to compute predictions.
        """
        num_batches = len(test_data) // batch_size
        predictions = np.empty(0, dtype=int)

        for batch_idx in tqdm(range(num_batches + 1), desc="Processing batches"):
            start = batch_idx * batch_size
            end = (
                start + batch_size
                if (start + batch_size) < len(test_data)
                else len(test_data)
            )
            batch_predictions = self.knn_predict(
                test_data[start:end], train_data, train_labels, k
            )
            predictions = np.concatenate((predictions, batch_predictions))

        return predictions

    def knn_predict(self, test_data, train_data, train_labels, k=1):
        predictions = np.empty(len(test_data), dtype=train_labels.dtype)
        for i in tqdm(range(len(test_data)), desc="Predicting", leave=False):
            distances = distance.cdist([test_data[i]], train_data, "euclidean")
            nearest_indices = np.argsort(distances[0])[:k]
            nearest_labels = train_labels[nearest_indices]
            predictions[i] = np.bincount(nearest_labels).argmax()
        return predictions

    def cluster_by_class(self, num_clusters=64):
        template_list = []  # List to store all cluster centers
        label_list = []  # List to store labels corresponding to each cluster center

        for label in tqdm(range(self.num_classes), desc="Clustering classes"):
            class_data = self.train_data[self.train_labels == label]
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(class_data)

            template_list.extend(kmeans.cluster_centers_)
            label_list.extend([label] * num_clusters)

        # Convert lists to numpy arrays before returning
        templates = np.array(template_list)
        labels = np.array(label_list)

        return templates, labels

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

    def get_error_rate(self, predictions):
        return np.mean(predictions != self.test_labels) * 100
