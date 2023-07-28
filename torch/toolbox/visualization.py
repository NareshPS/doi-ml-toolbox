"""Utilities to visualize inputs and predictions.
"""
import torch

from typing import List, Tuple
from matplotlib import pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix as plot_cm
from torch.utils.data import Dataset


def plot_image_classification(
    items: List,
    targets: List,
    predictions: List,
    class_names: List[str],
    rows: int = 4,
    item_size: float = 1,
    channel_first: bool = True,
    title: str = "Predictions",
):
    """Plots image classification results with targets and predictions."""

    # 1. Compute the number of columns based on the required number of rows.
    cols = (len(items) + rows - 1) // rows

    # 2. Compute the figsize based on the given item_size.
    figsize = (min(int(cols * item_size), 15), min(int(rows * item_size), 15))

    # 3. Create a figure to place the plots
    fig = plt.figure(figsize=figsize)

    # 4. Loop through the list of items
    for i, sample in enumerate(items):
        plt.subplot(rows, cols, i + 1)

        # 5. Translate the image to channel_last format
        sample = sample.permute(1, 2, 0) if channel_first else sample

        # 6. Plot the image
        plt.imshow(sample)

        # 7. Extract Predicted and True Label Names
        pred_label = class_names[predictions[i]]
        truth_label = class_names[targets[i]]

        # 8. Compose a title for the plot
        title_text = f"Pred: {predictions[i]} ({pred_label}) | Truth: {targets[i]} ({truth_label})"

        # 9. Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")  # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r")  # red text if wrong

        # 10. Remove axes
        plt.axis(False)

    fig.suptitle(title)


def plot_confusion_matrix(
    dataset: Dataset,
    preds: torch.Tensor,
    class_names: List[str],
    figsize: Tuple[float, float] = (10, 7),
):
    """Plots a confusion matrix for the dataset and the predictions."""
    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    confmat_tensor = confmat(preds=preds, target=torch.tensor(dataset.targets))

    fig, ax = plot_cm(
        conf_mat=confmat_tensor.numpy(),
        class_names=[
            f"{class_id} ({class_name})"
            for class_id, class_name in enumerate(class_names)
        ],
        figsize=(10, 7),
    )
