"""
Configuration constants for the Fashion-MNIST training script.

This file defines global constants for training that are shared across different
modules of the project, such as batch size, number of workers, and max epochs.

These parameters are primarily used when running the training script directly
from the terminal (`train.py`). The Gradio application provides its own UI controls
for these settings, so these values are not used when launching the app.
"""
# Experimental params
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_EPOCHS = 5