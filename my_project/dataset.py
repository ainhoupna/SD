import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import pytorch_lightning as pl


class FashionMNISTCSVDataset(Dataset):
    """
    Custom Dataset for Fashion-MNIST stored in CSV format.

    Each row in the CSV file contains:
    - column 0: label (0–9, corresponding to a clothing category)
    - columns 1–784: flattened pixel values (28x28 grayscale image)

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing the dataset.
    transform : callable, optional
        Transformation to apply to the images (e.g., `transforms.ToTensor()`).

    Examples
    --------
    >>> ds = FashionMNISTCSVDataset("data/raw/fashion-mnist_train.csv")
    >>> len(ds)
    60000
    >>> img, label = ds[0]
    >>> img.shape
    torch.Size([1, 28, 28])
    """

    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_data = self.data_frame.iloc[idx, 1:].values.astype(np.uint8)
        label = self.data_frame.iloc[idx, 0]
        image = image_data.reshape(28, 28)
        if self.transform:
            image = np.expand_dims(image, axis=-1)
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class FashionMNISTCSVDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Fashion-MNIST dataset (CSV format).

    This module wraps train, validation, and test datasets
    into corresponding DataLoaders for Lightning.

    Parameters
    ----------
    train_csv : str
        Path to the training CSV file.
    test_csv : str
        Path to the test CSV file.
    batch_size : int, optional (default=64)
        Batch size for training and evaluation.
    num_workers : int, optional (default=8)
        Number of worker processes for data loading.

    Examples
    --------
    >>> dm = FashionMNISTCSVDataModule(
    ...     train_csv="data/raw/fashion-mnist_train.csv",
    ...     test_csv="data/raw/fashion-mnist_test.csv"
    ... )
    >>> dm.setup("fit")
    >>> len(dm.train_dataloader())
    938
    """

    def __init__(self, train_csv, test_csv, batch_size=64, num_workers=8):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        """
        Placeholder for dataset preparation.

        This method is part of the LightningDataModule interface,
        but here it does nothing since data is already available.

        Returns
        -------
        None
        """
        pass

    def setup(self, stage=None):
        """
        Setup datasets for training, validation, and testing.

        Parameters
        ----------
        stage : str or None, optional
            Current stage ("fit", "test", or None).
            If None, both training/validation and test datasets are initialized.

        Returns
        -------
        None
        """

        if stage == "fit" or stage is None:
            self.train_ds = FashionMNISTCSVDataset(
                self.train_csv, transform=self.transform
            )
            self.val_ds = FashionMNISTCSVDataset(
                self.test_csv, transform=self.transform
            )
        if stage == "test" or stage is None:
            self.test_ds = FashionMNISTCSVDataset(
                self.test_csv, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1000, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1000, num_workers=self.num_workers)
