## Description

This project implements an image classification model based on the Fashion MNIST dataset, using PyTorch Lightning to structure the code in a modular and scalable way.
The dataset is in CSV format, with 28x28 pixel grayscale images and their respective labels. The goal is to train a simple convolutional network that classifies images into 10 different clothing categories.

The goal is to train a simple convolutional neural network (CNN) to classify images into the following categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.
## Project Structure

├── data/               # Original and processed datasets
│   └── raw/            # Fashion-MNIST CSV and IDX files
├── models/             # Saved models and checkpoints
├── lightning_logs/     # PyTorch Lightning logs and training runs
├── my_project/         # Source code
│   ├── config.py       # Training and configuration parameters
│   ├── dataset.py      # Custom Dataset and DataModule
│   ├── model.py        # PyTorch Lightning model definition
│   ├── train.py        # Training and evaluation script
│   ├── plots.py        # Visualization utilities
│   └── EDA.ipynb       # Exploratory Data Analysis notebook
├── reports/            # Reports and analysis
│   ├── figures/        # Generated plots and metrics
│   └── Model_Performance.pdf # Model performance report
├── README.md           # Project documentation
├── pyproject.toml      # Dependency management (Poetry)

## Installation
To install dependencies and prepare the environment with uv, run the following commands in the terminal:
1. Download and install dependencies: curl -sSf https://uv.io/install.sh | sh
2. Initialize the environment: uv init
3. Sync dependencies and environment: uv sync

## Training and Evaluation
Run in the terminal: uv run python -m my_project.train

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
