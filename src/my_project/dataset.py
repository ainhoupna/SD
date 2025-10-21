import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class FashionMNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Fashion-MNIST dataset.
    It handles the downloading, splitting, and loading of the data.
    """

    def __init__(self, data_dir: str = "data/", batch_size: int = 128, val_split: float = 0.2, num_workers: int = 4):
        """
        Args:
            data_dir (str): Directory where the data will be downloaded/stored.
            batch_size (int): The batch size for the data loaders.
            val_split (float): The fraction of the training data to use for validation.
            num_workers (int): Number of subprocesses to use for data loading.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers # Store num_workers
        self.val_split = val_split
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        )
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_val_dataset = None # For data exploration tab

    def prepare_data(self):
        """
        Downloads the Fashion-MNIST dataset if it's not already present.
        This method is called only on a single GPU/process.
        """
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        """
        Assigns train/val/test datasets for dataloaders.
        This method is called on every GPU.
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_val_dataset = datasets.FashionMNIST(self.data_dir, train=True, transform=self.transform)
            n_samples = len(self.train_val_dataset)
            n_val = int(self.val_split * n_samples)
            n_train = n_samples - n_val
            self.train_dataset, self.val_dataset = random_split(self.train_val_dataset, [n_train, n_val])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = datasets.FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)