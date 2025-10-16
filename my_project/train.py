from my_project.config import TRAIN_CSV_PATH, TEST_CSV_PATH
from my_project.dataset import FashionMNISTCSVDataModule
from my_project.model import Net
from my_project.plots import evaluate_and_plot, plot_curves_from_csvlogger
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# Experimental params
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_EPOCHS = 5
LEARNING_RATE = 1e-3


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
    None

    Examples
    --------
    Run training from the command line:

    >>> python -m my_project.train
    """

    data_module = FashionMNISTCSVDataModule(
        train_csv=TRAIN_CSV_PATH,
        test_csv=TEST_CSV_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    net = Net(num_filters=32, hidden_size=64, lr=LEARNING_RATE)

    logger = CSVLogger("lightning_logs", name="fashion_model")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=logger,
    )

    trainer.fit(net, datamodule=data_module)
    trainer.test(net, datamodule=data_module)

    artifacts = evaluate_and_plot(net, data_module, out_dir="reports/figures")
    plot_curves_from_csvlogger(logger.log_dir)

    print(f"Test accuracy: {artifacts['test_accuracy']:.4f}")
    print("Saved figures:")
    for k, v in artifacts.items():
        if k != "test_accuracy":
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
