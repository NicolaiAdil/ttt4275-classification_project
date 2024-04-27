import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define global font sizes
SIZE_LABEL = 28 # Define the font size for the labels
SIZE_TITLE = 32 # Define the font size for the title
SIZE_AXIS = 25 # Define the font size for the axis ticks

def plot_characteristics_separability(
    setosa_matrix: np.ndarray,
    versicolor_matrix: np.ndarray,
    virginica_matrix: np.ndarray,
    iris_feature_names: list,
) -> None:
    """
    Plots the characteristics of the iris dataset for each class.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))

    # Setosa
    axs[0].scatter(setosa_matrix[:, 0], setosa_matrix[:, 1], color="r", label="Setosa")
    # Versicolor
    axs[0].scatter(
        versicolor_matrix[:, 0], versicolor_matrix[:, 1], color="g", label="Versicolor"
    )
    # Virginica
    axs[0].scatter(
        virginica_matrix[:, 0], virginica_matrix[:, 1], color="b", label="Virginica"
    )

    axs[0].set_xlabel(iris_feature_names[0], fontsize=SIZE_LABEL)
    axs[0].set_ylabel(iris_feature_names[1], fontsize=SIZE_LABEL)
    axs[0].set_title("Sepal Characteristics of Iris Dataset", fontsize=SIZE_TITLE)
    axs[0].legend(fontsize=SIZE_LABEL)
    axs[0].tick_params(axis='both', which='major', labelsize=SIZE_AXIS)

    # Plot petal length and width
    axs[1].scatter(setosa_matrix[:, 2], setosa_matrix[:, 3], color="r", label="Setosa")
    axs[1].scatter(
        versicolor_matrix[:, 2], versicolor_matrix[:, 3], color="g", label="Versicolor"
    )
    axs[1].scatter(
        virginica_matrix[:, 2], virginica_matrix[:, 3], color="b", label="Virginica"
    )

    axs[1].set_xlabel(iris_feature_names[2], fontsize=SIZE_LABEL)
    axs[1].set_ylabel(iris_feature_names[3], fontsize=SIZE_LABEL)
    axs[1].set_title("Petal Characteristics of Iris Dataset", fontsize=SIZE_TITLE)
    axs[1].legend(fontsize=SIZE_LABEL)
    axs[1].tick_params(axis='both', which='major', labelsize=SIZE_AXIS)

    plt.tight_layout()
    plt.show()

def plot_error_rate(error_rate_vector, error_rate_test_vector, title):
    plt.figure(figsize=(10, 8))
    plt.plot(error_rate_vector, label="Training data")
    plt.plot(error_rate_test_vector, label="Test data")
    plt.xlabel("Iteration", fontsize=SIZE_LABEL)
    plt.ylabel("Error rate [%]", fontsize=SIZE_LABEL)
    plt.title(f"Error rate of the training and test data\n{title}", fontsize=SIZE_TITLE)
    plt.legend(fontsize=SIZE_LABEL)
    plt.xticks(fontsize=SIZE_AXIS)
    plt.yticks(fontsize=SIZE_AXIS)
    plt.show()

def plot_confusion_matrix(classifier, iris, train_or_test, title, error_rate_train = -1, error_rate_test = -1):
    confusion_matrix = classifier.get_confusion_matrix(train_or_test=train_or_test)
    error_rate = error_rate_train if train_or_test == "train" else error_rate_test
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        annot_kws={"size": SIZE_AXIS},
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )
    plt.xlabel("Predicted", fontsize=SIZE_LABEL)
    plt.ylabel("Actual", fontsize=SIZE_LABEL)
    if error_rate != -1:
        plt.title(f"Confusion Matrix for {train_or_test} data\n{title}\nError rate: {error_rate * 100:.2f}%", fontsize=SIZE_TITLE)
    else:
        plt.title(f"Confusion Matrix for {train_or_test} data\n{title}", fontsize=SIZE_TITLE)
    plt.xticks(fontsize=SIZE_AXIS)
    plt.yticks(fontsize=SIZE_AXIS)

    # Adjust colorbar label and tick font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Scale', fontsize=SIZE_LABEL)
    cbar.ax.tick_params(labelsize=SIZE_AXIS)  # Set tick label size

    plt.show()

def plot_characteristics_separability_histogram(data, labels, features, class_names):
    # Increasing the figure size to better accommodate larger fonts and prevent squishing
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 24))  # Adjusted figsize

    colors = ["blue", "orange", "green"]

    # Plotting histograms
    for i, feature in enumerate(features):
        ax = axes[i]
        for j, class_name in enumerate(class_names):
            class_mask = labels == j
            ax.hist(
                data[class_mask, i],
                bins=10,
                color=colors[j],
                alpha=0.7,
                label=f"{class_name} ({feature})",
            )

        ax.set_title(f"Histogram of {feature}", fontsize=SIZE_TITLE)
        ax.set_xlabel("Measurement [cm]", fontsize=SIZE_LABEL-3)
        ax.set_ylabel("Frequency", fontsize=SIZE_LABEL)
        ax.legend(fontsize=(SIZE_LABEL-8))
        ax.tick_params(axis='both', labelsize=SIZE_AXIS)

    # Adjust layout to prevent overlap, taking care not to compress the plot
    fig.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.5)  # Adjust top margin and the space between plots
    plt.show()

