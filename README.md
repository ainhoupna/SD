## Description

This project implements an image classification model based on the Fashion MNIST dataset, using PyTorch Lightning to structure the code in a modular and scalable way.
The dataset is in CSV format, with 28x28 pixel grayscale images and their respective labels. The goal is to train a simple convolutional network that classifies images into 10 different clothing categories.

## Project Structure

data/ # Original and processed data
 └── raw/ # Original CSV files (fashion-mnist_train.csv, fashion-mnist_test.csv)
models/ # Trained models, checkpoints, and saved weights
src/
 └── my_project/ # Project source code
     ├── dataset.py # Dataset and DataModule
     ├── model.py # PyTorch Lightning model
     ├── train.py # Main training and evaluation script
     └── config.py # Project configurations and parameters
tests/ # Tests for the project
LICENSE # Project license
README.md # Project documentation
pyproject.toml # Dependency management and Poetry configuration
uv.lock # Dependency lock file 

## Installation
To install dependencies and prepare the environment with uv, run the following commands in the terminal:
1. Download and install dependencies: curl -sSf https://uv.io/install.sh | sh
2. Initialize the environment: uv init
3. Sync dependencies and environment: uv sync

## Training and Evaluation
To run the training and evaluation script, first install the package in editable mode:

```bash
uv pip install -e .
```

Then, run the following command in the terminal:

```bash
my_project
```

## Building and Publishing to PyPI
To build and publish the package to PyPI, follow these steps:

1.  **Install build tools:**
    ```bash
    uv pip install build twine
    ```
2.  **Build the package:**
    ```bash
    python -m build
    ```
3.  **Publish to PyPI:**
    ```bash
    twine upload dist/*
    ```
    You will be prompted for your PyPI username and password.


## Technical Details
-Dataset: Custom Dataset that reads images from CSV and applies transformations.
-Model: Simple CNN with one convolutional layer, pooling, and fully connected layers.
-Training: Implemented with PyTorch Lightning to facilitate handling epochs, performance, and metrics.
-Configuration: Parameters such as batch size, paths, epochs defined in config.py.
-Optimization: Adam with CrossEntropyLoss.

## Reports & Visualizations
The project includes detailed reports and visualizations:
- **Confusion matrix**
- **Per-class accuracy**
- **Calibration curve**
- **Misclassified image grids**
- **Pixel distribution analysis**

All outputs are stored in the `reports/` folder.

## Contact
-delrey.132148@e.unavarra.es
-goicoechea.128710@e.unavarra.es
-haddad.179806@e.unavarra.es
