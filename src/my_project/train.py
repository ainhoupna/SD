from my_project.dataset import FashionMNISTDataModule
from my_project.model import Net
from my_project.plots import evaluate_and_plot
import pytorch_lightning as pl

"""
Train and evaluate the Fashion-MNIST model.

This script can be run from the command line to:
1. Initialize the DataModule and model with default or specified parameters.
2. Run the training process for a fixed number of epochs.
3. Evaluate the trained model on the test set.
4. Save evaluation figures (like confusion matrix) in `reports/figures/`.

Generated Artifacts
-------------------
When you run this script, the following directories and files may be created:

- `data/`:
  - Contains the downloaded Fashion-MNIST dataset.

- `models/lightning_logs/`:
  - Stores logs and checkpoints from PyTorch Lightning during training.

- `reports/figures/`:
  - Contains output visualizations from the evaluation step, such as `confusion_matrix.png`, `per_class_accuracy.png`, etc.

Examples
--------
Run training from the command line:

>>> python -m my_project.train
"""

# Experimental params
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_EPOCHS = 5

def main():
    """
    Train and evaluate the Fashion-MNIST model.

    This script:
    1. Initializes the DataModule and model.
    2. Runs training for a fixed number of epochs.
    3. Evaluates on the test set.
    4. Saves evaluation figures in `reports/figures/`.

    Returns
    -------
    The main training and evaluation function.
    """

    data_module = FashionMNISTDataModule(
        data_dir="data/",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        # The num_workers parameter is now accepted by FashionMNISTDataModule
    )

    net = Net(num_filters=32, hidden_size=64)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        default_root_dir="models/lightning_logs",
    )

    trainer.fit(net, datamodule=data_module)
    trainer.test(net, datamodule=data_module)

    artifacts = evaluate_and_plot(net, data_module, out_dir="reports/figures")
    print(f"Test accuracy: {artifacts['test_accuracy']:.4f}")
    print("Saved figures:")
    for k, v in artifacts.items():
        if k != "test_accuracy":
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
