"""
Configuration constants for the Fashion-MNIST CSV project.

This file holds global constants such as dataset paths and worker counts.
Modify experiment-specific hyperparameters (e.g., batch size, epochs) directly
in the training scripts (`train.py`, `train_expX.py`) to allow reproducibility
of multiple experiments.
"""

TRAIN_CSV_PATH = "./data/raw/fashion-mnist_train.csv"
TEST_CSV_PATH = "./data/raw/fashion-mnist_test.csv"
NUM_WORKERS = 8
