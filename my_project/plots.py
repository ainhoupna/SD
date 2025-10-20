# my_project/plot.py
"""
Utilities to evaluate a trained model and generate performance visualizations.
- evaluate_and_plot(...) : runs eval on a dataloader and saves:
    * confusion_matrix.png
    * per_class_accuracy.png
    * misclassified_grid.png
- plot_curves_from_csvlogger(...) : plots curves from Lightning's CSVLogger
    * train_loss_[step|epoch].png
    * val_acc_epoch.png (if logged during validation)
Outputs are saved under reports/figures/ by default.

This module does NOT modify your model architecture.
"""

import os
import math
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist


# Fashion-MNIST class names (0..9)
FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _get_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over a dataloader and return predictions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader for evaluation (test/validation).

    Returns
    -------
    tuple
        (y_true, y_pred, y_prob_max):
        - y_true (ndarray[int]): Ground-truth labels.
        - y_pred (ndarray[int]): Predicted labels.
        - y_prob_max (ndarray[float]): Max softmax confidence per sample.
    """

    model.eval()
    device = _get_device(model)

    all_true: List[int] = []
    all_pred: List[int] = []
    all_conf: List[float] = []

    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        all_true.extend(_to_numpy(yb))
        all_pred.extend(_to_numpy(pred))
        all_conf.extend(_to_numpy(conf))

    return (
        np.asarray(all_true, dtype=np.int64),
        np.asarray(all_pred, dtype=np.int64),
        np.asarray(all_conf, dtype=np.float32),
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: str,
    normalize: bool = True,
) -> None:
    """
    Plot and save a confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list of str
        Class names corresponding to label indices.
    out_path : str
        File path to save the plot.
    normalize : bool, optional (default=True)
        If True, normalize counts to percentages.
    """

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="coolwarm")
    plt.title("Confusion Matrix " if normalize else "Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(
                j,
                i,
                txt,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: str,
) -> None:
    """
    Plot and save per-class accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list of str
        Class names corresponding to label indices.
    out_path : str
        File path to save the plot.
    """

    num_classes = len(class_names)
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        total[t] += 1
        if t == p:
            correct[t] += 1

    acc = np.divide(correct, np.maximum(total, 1), where=total > 0)

    plt.figure(figsize=(9, 4))
    plt.bar(np.arange(num_classes), acc)
    plt.ylim(0, 1)
    plt.xticks(np.arange(num_classes), class_names, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Per-class Accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _denormalize_img(img: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor image back to NumPy format.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor (CHW or HW).

    Returns
    -------
    np.ndarray
        Image array suitable for plotting.
    """

    if img.dim() == 3 and img.shape[0] == 1:
        img_np = img.squeeze(0).detach().cpu().numpy()
        return img_np
    # If already HxW
    return img.detach().cpu().numpy()


@torch.no_grad()
def plot_misclassified_grid(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    out_path: str,
    max_examples: int = 16,
) -> None:
    """
    Plot a grid of misclassified samples.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader for evaluation (test/validation).
    class_names : list of str
        Class names corresponding to label indices.
    out_path : str
        File path to save the plot.
    max_examples : int, optional (default=16)
        Maximum number of misclassified samples to show.
    """
    model.eval()
    device = _get_device(model)

    images = []
    labels_true = []
    labels_pred = []

    # Collect up to max_examples misclassified samples
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)

        mism = pred != yb
        if mism.any():
            mis_idx = torch.nonzero(mism).flatten()
            for idx in mis_idx:
                images.append(xb[idx].cpu())
                labels_true.append(int(yb[idx].cpu()))
                labels_pred.append(int(pred[idx].cpu()))
                if len(images) >= max_examples:
                    break
        if len(images) >= max_examples:
            break

    if len(images) == 0:
        # Nothing to plot
        fig = plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No misclassified samples found.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close()
        return

    # Determine grid size
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))

    plt.figure(figsize=(cols * 1.5, rows * 1.5))
    for i, img in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(_denormalize_img(img), cmap="gray")
        true_name = class_names[labels_true[i]]
        pred_name = class_names[labels_pred[i]]
        ax.set_title(f"{true_name} → {pred_name}", fontsize=8)
        ax.axis("off")

    plt.suptitle("Misclassified Samples", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def evaluate_and_plot(
    model: torch.nn.Module,
    datamodule,
    out_dir: str = "reports/figures",
) -> dict:

    _ensure_dir(out_dir)

    # Use the datamodule's test dataloader (Lightning-style)
    test_loader = datamodule.test_dataloader()

    # 1) Predictions + basic accuracy
    y_true, y_pred, y_conf = evaluate_model(model, test_loader)
    test_acc = float((y_true == y_pred).mean())

    # 2) Confusion matrix (normalized)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, FASHION_CLASSES, cm_path, normalize=True)

    # 3) Per-class accuracy bars
    pca_path = os.path.join(out_dir, "per_class_accuracy.png")
    plot_per_class_accuracy(y_true, y_pred, FASHION_CLASSES, pca_path)

    # 4) Misclassified samples grid
    mis_path = os.path.join(out_dir, "misclassified_grid.png")
    plot_misclassified_grid(
        model, test_loader, FASHION_CLASSES, mis_path, max_examples=6
    )

    # 5) Calibration (reliability) curve
    calib_path = os.path.join(out_dir, "calibration_curve.png")
    y_correct = (y_true == y_pred).astype(int)
    plot_calibration_curve(y_correct, y_conf, calib_path, n_bins=10)

    return {
        "test_accuracy": test_acc,
        "confusion_matrix": cm_path,
        "per_class_accuracy": pca_path,
        "misclassified_grid": mis_path,
        "calibration_curve": calib_path,
    }


def plot_calibration_curve(y_true, y_prob, out_path, n_bins: int = 10):

    # Calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_curves_from_csvlogger(
    csv_log_dir: str,
    out_dir: str = "reports/figures",
    train_loss_keys: Optional[List[str]] = None,
    val_acc_keys: Optional[List[str]] = None,
) -> List[str]:
    """
    If you use Lightning's CSVLogger, this parses the `metrics.csv` file and
    produces line plots for:
      - train loss (step/epoch)
      - val accuracy (epoch)

    Usage in train.py:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger("logs", name="fashion")
        trainer = pl.Trainer(..., logger=logger)

    Then call:
        plot_curves_from_csvlogger(logger.log_dir)

    Args:
        csv_log_dir: e.g., "logs/fashion/version_0"
        train_loss_keys: possible metric column names for train loss
        val_acc_keys: possible metric column names for val acc

    Returns:
        list of saved figure paths
    """
    _ensure_dir(out_dir)
    metrics_path = os.path.join(csv_log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return []

    df = pd.read_csv(metrics_path)

    # Reasonable defaults; adjust if you log different names
    if train_loss_keys is None:
        # Lightning often stores as "train_loss_step" and/or "train_loss_epoch"
        train_loss_keys = ["train_loss_step", "train_loss_epoch", "loss", "train_loss"]
    if val_acc_keys is None:
        # You already log 'val_acc' in your model
        val_acc_keys = ["val_acc", "val_acc_epoch"]

    saved = []

    # Train loss vs step/epoch
    for key in train_loss_keys:
        if key in df.columns:
            # Prefer 'step' index if present, else 'epoch'
            if "step" in df.columns and not df["step"].isna().all():
                x = df["step"]
                x_label = "Step"
            else:
                x = df["epoch"] if "epoch" in df.columns else np.arange(len(df))
                x_label = "Epoch"

            plt.figure(figsize=(6, 4))
            plt.plot(x, df[key])
            plt.xlabel(x_label)
            plt.ylabel("Train Loss")
            plt.title(f"{key} over {x_label.lower()}")
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{key}.png")
            plt.savefig(out_path, dpi=160)
            plt.close()
            saved.append(out_path)
            # Only plot first that exists (avoid duplicates)
            break

    # Validation accuracy vs epoch
    for key in val_acc_keys:
        if key in df.columns:
            x = df["epoch"] if "epoch" in df.columns else np.arange(len(df))
            plt.figure(figsize=(6, 4))
            plt.plot(x, df[key])
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title(f"{key} over epochs")
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{key}.png")
            plt.savefig(out_path, dpi=160)
            plt.close()
            saved.append(out_path)
            break

    return saved


def plot_learning_curves_from_df(
    df: pd.DataFrame,
) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    """
    Parses a metrics DataFrame and produces line plots for train loss and val accuracy.

    Args:
        df (pd.DataFrame): DataFrame with metrics (e.g., from CSVLogger).

    Returns:
        A tuple of (train_loss_figure, val_acc_figure). Figures can be None.
    """
    train_loss_fig, val_acc_fig = None, None

    # Train loss vs step/epoch
    train_loss_keys = ["train_loss_step", "train_loss_epoch", "loss", "train_loss"]
    val_acc_keys = ["val_acc", "val_acc_epoch"]

    for key in train_loss_keys:
        if key in df.columns and not df[key].isna().all():
            series = df[key].dropna()
            # Prefer 'step' index if present, else 'epoch'
            if "step" in df.columns and not df["step"].isna().all():
                x = df.loc[series.index, "step"]
                x_label = "Step"
            else:
                x = df.loc[series.index, "epoch"]
                x_label = "Epoch"

            train_loss_fig = plt.figure(figsize=(6, 4))
            plt.plot(x, series)
            plt.xlabel(x_label)
            plt.ylabel("Train Loss")
            plt.title(f"Training Loss vs. {x_label}")
            plt.tight_layout()
            # Only plot first that exists
            break

    # Validation accuracy vs epoch
    for key in val_acc_keys:
        if key in df.columns and not df[key].isna().all():
            # Filter to get only the steps where validation was run
            series = df[key].dropna()
            x = df.loc[series.index, "epoch"]
            val_acc_fig = plt.figure(figsize=(6, 4))
            plt.plot(x, series, marker='o', linestyle='-')
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title("Validation Accuracy vs. Epoch")
            plt.ylim(0, 1)
            plt.tight_layout()
            # Only plot first that exists
            break

    # Close figures if they were not created to avoid empty plots
    if train_loss_fig is None: plt.close()
    if val_acc_fig is None: plt.close()

    return train_loss_fig, val_acc_fig


def plot_class_distribution(
    df: pd.DataFrame, class_names: List[str], out_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the distribution of classes in a dataset.

    Args:
        df (pd.DataFrame): DataFrame with a 'label' column.
        class_names (List[str]): List of class names.
        out_path (Optional[str], optional): If provided, saves the plot. Defaults to None.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    class_counts = df["label"].value_counts().sort_index()
    total_samples = class_counts.sum()

    fig = plt.figure(figsize=(9, 4))
    bars = plt.bar(class_counts.index, class_counts.values)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylabel("Number of Samples")
    plt.title("Dataset Class Distribution")

    # Add count and percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height}\n({height/total_samples:.1%})', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if out_path:
        _ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=160)
        plt.close()

    return fig


def get_sample_images_for_gallery(
    df: pd.DataFrame, class_names: List[str], n_samples: int = 20
) -> List[Tuple[np.ndarray, str]]:
    """
    Gets a list of random sample images and their labels for a Gradio Gallery.

    Args:
        df (pd.DataFrame): DataFrame with image data.
        class_names (List[str]): List of class names.
        n_samples (int, optional): Number of samples to retrieve. Defaults to 20.

    Returns:
        List[Tuple[np.ndarray, str]]: A list of tuples, each containing an image and its label.
    """
    samples = []
    if len(df) > n_samples:
        df_sample = df.sample(n=n_samples)
    else:
        df_sample = df

    for _, row in df_sample.iterrows():
        label = class_names[int(row["label"])]
        image_data = row.iloc[1:].values.astype(np.uint8)
        image = image_data.reshape(28, 28)
        # The label is now just plain text
        samples.append((image, label))
    return samples


def plot_class_correlation_dendrogram(
    df: pd.DataFrame, class_names: List[str], out_path: Optional[str] = None
) -> plt.Figure:
    """
    Calculates and plots a dendrogram showing the similarity between the
    average image of each class.

    Args:
        df (pd.DataFrame): DataFrame with image data and labels.
        class_names (List[str]): List of class names.
        out_path (Optional[str], optional): If provided, saves the plot. Defaults to None.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    mean_images = []
    for i in range(len(class_names)):
        # Filter for the class, get pixel data, and calculate the mean image
        mean_img = df[df["label"] == i].iloc[:, 1:].mean().values
        mean_images.append(mean_img)

    # Convert list of mean images to a matrix
    mean_images_matrix = np.vstack(mean_images)

    # Perform hierarchical clustering
    linked = sch.linkage(mean_images_matrix, method="ward")

    fig = plt.figure(figsize=(9, 4))
    sch.dendrogram(linked, orientation="top", labels=class_names, leaf_rotation=90)
    plt.ylabel("Euclidean Distance (between mean images)")
    plt.title("Class Similarity Dendrogram")
    plt.tight_layout()

    if out_path:
        _ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=160)
        plt.close()

    return fig
