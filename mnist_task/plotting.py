import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Global style variables
TITLE_SIZE = 32
LABEL_SIZE = 22
AXIS_TITLE_SIZE = 25

def plot_confusion_matrix(confusion_matrix, num_classes, title, train_or_test="test"):
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        annot_kws={"size": AXIS_TITLE_SIZE},
        cmap="Blues",
        xticklabels=[str(i) for i in range(num_classes)],
        yticklabels=[str(i) for i in range(num_classes)]
    )
    plt.xlabel("Predicted", fontsize=AXIS_TITLE_SIZE)
    plt.ylabel("Actual", fontsize=AXIS_TITLE_SIZE)

    plt.xticks(fontsize=AXIS_TITLE_SIZE)
    plt.yticks(fontsize=AXIS_TITLE_SIZE)

    plt.title(f"Confusion Matrix for {train_or_test.capitalize()} Data\n{title}", fontsize=TITLE_SIZE)

    # Adjust colorbar label and tick font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Scale', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)  # Set tick label size

    plt.show()

def plot_histogram(test_labels, train_labels, title="Histogram of class distribution"):
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    unique_train, counts_train = np.unique(train_labels, return_counts=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.bar(unique_test, counts_test, color="blue")
    ax1.set_title(f"Test Data Class Distribution. N = {len(test_labels)}", fontsize=TITLE_SIZE)
    ax1.set_xlabel("Class", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Frequency", fontsize=LABEL_SIZE)
    ax1.tick_params(axis='x', labelsize=AXIS_TITLE_SIZE)
    ax1.tick_params(axis='y', labelsize=AXIS_TITLE_SIZE)

    ax2.bar(unique_train, counts_train, color="green")
    ax2.set_title(f"Training Data Class Distribution. N = {len(train_labels)}", fontsize=TITLE_SIZE)
    ax2.set_xlabel("Class", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Frequency", fontsize=LABEL_SIZE)
    ax2.tick_params(axis='x', labelsize=AXIS_TITLE_SIZE)
    ax2.tick_params(axis='y', labelsize=AXIS_TITLE_SIZE)

    plt.suptitle(title, fontsize=TITLE_SIZE + 2)  # slightly larger title for the figure
    plt.tight_layout()
    plt.show()

def display_classification(images, labels, predictions, num_images=5, title="Classification Results"):
    correct_indices = np.where(labels == predictions)[0]
    incorrect_indices = np.where(labels != predictions)[0]
    num_correct_display = min(len(correct_indices), num_images)
    num_incorrect_display = min(len(incorrect_indices), num_images)
    fig, axs = plt.subplots(nrows=2, ncols=max(num_correct_display, num_incorrect_display), figsize=(15, 6))
    for idx in range(num_correct_display):
        ax = axs[0, idx]
        image_idx = correct_indices[idx]
        ax.imshow(images[image_idx], cmap="gray")
        ax.set_title(f"Prediction: {predictions[image_idx]}\n (True: {labels[image_idx]})", fontsize=LABEL_SIZE)
        ax.axis("off")
    for idx in range(num_incorrect_display):
        ax = axs[1, idx]
        image_idx = incorrect_indices[idx]
        ax.imshow(images[image_idx], cmap="gray")
        ax.set_title(f"Prediction: {predictions[image_idx]}\n(True: {labels[image_idx]})", fontsize=LABEL_SIZE)
        ax.axis("off")
    plt.tight_layout(pad=3.0)
    plt.suptitle(title, fontsize=TITLE_SIZE)
    plt.show()

def plot_templates(templates, labels, nClusters, nTemplates=6, nClasses=10):
    fig, axes = plt.subplots(nClasses, nTemplates, figsize=(12, 20))
    for i in range(nClasses):
        class_templates = templates[labels == i]
        for j in range(nTemplates):
            ax = axes[i, j]
            if j < len(class_templates):
                ax.imshow(class_templates[j].reshape(28, 28), cmap="gray")
                ax.set_title(f"Class {i} Template {j+1}", fontsize=LABEL_SIZE)
            ax.axis("off")
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95, hspace=0.6, wspace=0.3)
    fig.suptitle(f"The first {nTemplates} of {nClusters} Cluster Centers for Each Class", fontsize=TITLE_SIZE, y=0.95)
    plt.show()

def plot_examples(images, labels, num_classes):
    _, axes = plt.subplots(2, num_classes // 2, figsize=(10, 4))
    for i in range(num_classes):
        row = i // 5
        col = i % 5
        idx = np.where(labels == i)[0][0]
        ax = axes[row, col]
        ax.imshow(images[idx], cmap='gray')
        ax.set_title(f'Class {i}', fontsize=TITLE_SIZE)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
