import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix, num_classes, train_or_test="test"):
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(i) for i in range(num_classes)],
        yticklabels=[str(i) for i in range(num_classes)]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {train_or_test.capitalize()} Data")
    plt.show()