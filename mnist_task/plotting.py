import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    confusion_matrix, num_classes, error_rate, k, train_or_test="test"
):
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(i) for i in range(num_classes)],
        yticklabels=[str(i) for i in range(num_classes)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"Confusion Matrix for {train_or_test.capitalize()} Data.\nK = {k}\nError rate : {error_rate:.2f}%"
    )
    plt.show()


def plot_histogram(test_labels, train_labels, title="Histogram of class distribution"):
    # Extract labels and their counts for the histograms
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    unique_train, counts_train = np.unique(train_labels, return_counts=True)

    # Creating a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6)
    )  # 1 row, 2 columns, figure size 12x6 inches

    # Plotting the test data histogram
    ax1.bar(unique_test, counts_test, color="blue")
    ax1.set_title(f"Test Data Class Distribution. N = {len(test_labels)}")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Frequency")

    # Plotting the training data histogram
    ax2.bar(unique_train, counts_train, color="green")
    ax2.set_title(f"Training Data Class Distribution. N = {len(train_labels)}")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Frequency")

    # Setting a main title for the figure
    plt.suptitle(title)

    plt.show()


def display_classification(
    images, labels, predictions, num_images=5, title="Classification Results"
):
    """
    Displays a specified number of correctly and incorrectly classified images in separate subplots using matplotlib.
    Args:
        images: The array of images (reshaped into original dimensions if needed).
        labels: The actual labels of the images.
        predictions: The predicted labels of the images.
        num_images: Number of correctly and incorrectly classified images to display.
        title: The figure title.
    """
    correct_indices = np.where(labels == predictions)[0]
    incorrect_indices = np.where(labels != predictions)[0]

    # Ensure we do not exceed the number of available correct or incorrect indices
    num_correct_display = min(len(correct_indices), num_images)
    num_incorrect_display = min(len(incorrect_indices), num_images)

    fig, axs = plt.subplots(
        nrows=2, ncols=max(num_correct_display, num_incorrect_display), figsize=(15, 6)
    )  # 2 rows, one for correct and one for incorrect

    # Display correctly classified images
    for idx in range(num_correct_display):
        ax = axs[0, idx]
        image_idx = correct_indices[idx]
        ax.imshow(images[image_idx], cmap="gray")
        ax.set_title(f"Correct prediction: {predictions[image_idx]}")
        ax.axis("off")  # Hide axes

    # Display incorrectly classified images
    for idx in range(num_incorrect_display):
        ax = axs[1, idx]
        image_idx = incorrect_indices[idx]
        ax.imshow(images[image_idx], cmap="gray")
        ax.set_title(
            f"Incorrect prediction: {predictions[image_idx]}\n(True: {labels[image_idx]})"
        )
        ax.axis("off")  # Hide axes

    # If fewer plots in a row, turn off unused axes
    for idx in range(num_correct_display, axs.shape[1]):
        axs[0, idx].axis("off")
    for idx in range(num_incorrect_display, axs.shape[1]):
        axs[1, idx].axis("off")

    plt.tight_layout(pad=3.0)  # Add padding between plots
    plt.suptitle(title)
    plt.show()


def plot_templates(templates, labels, nTemplates=6, nClasses=10):
    fig, axes = plt.subplots(
        nClasses, nTemplates, figsize=(12, 20)
    )  # 10 rows for 10 classes

    # Iterate over each class
    for i in range(nClasses):
        # Select the templates for the current class
        class_templates = templates[labels == i]

        # Display up to nTemplates cluster centers for each class
        for j in range(nTemplates):
            ax = axes[i, j]
            if j < len(class_templates):  # Check if there's a template to display
                ax.imshow(class_templates[j].reshape(28, 28), cmap="gray")
                ax.set_title(
                    f"Class {i} Template {j+1}", fontsize=10
                )  # Optionally adjust font size for better fit
            ax.axis("off")  # Hide axes for all plots regardless

    # Set a common title for all subplots
    fig.suptitle(
        f"The first {nTemplates} K-Means Cluster Centers for Each Class",
        fontsize=16,
        y=0.95,
    )

    # Adjust layout parameters to avoid overlap and ensure everything fits
    plt.subplots_adjust(
        top=0.90, bottom=0.05, left=0.05, right=0.95, hspace=0.6, wspace=0.3
    )

    plt.show()
