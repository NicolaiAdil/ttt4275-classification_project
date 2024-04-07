from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load MNIST dataset
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print("MNIST dataset loaded.")

# Normalize pixel values
X = X / 255.0

# Split dataset into chunks
print("Splitting dataset into chunks...")
chunk_size = 1000
X_chunks = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
y_chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
print("Dataset split into chunks.")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7.0, random_state=0)

# Create a k-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)

# Train the classifier
print("Training the classifier...")
for i in range(len(X_chunks)):
    knn.fit(X_chunks[i], y_chunks[i])
print("Classifier trained.")

# Predict the labels for the test data
print("Predicting the labels for the test data...")
y_pred = knn.predict(X_test)
print("Labels predicted.")

# Compute the confusion matrix and error rate
cm = confusion_matrix(y_test, y_pred)
error_rate = (y_test != y_pred).mean()

print("Confusion Matrix:")
print(cm)
print("Error Rate:", error_rate)
