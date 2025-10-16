import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl


class Net(pl.LightningModule):
    """
    Simple convolutional neural network for Fashion-MNIST classification.

    Architecture
    ------------
    - Conv2d(1 → 16, kernel_size=3)
    - ReLU
    - MaxPool2d(kernel_size=2)
    - Flatten
    - Linear(16*13*13 → 32)  # Note: This will be configurable
    - ReLU
    - Linear(32 → 10)       # Note: This will be configurable

    Loss: CrossEntropyLoss

    Examples
    --------
    >>> model = Net()
    >>> x = torch.randn(8, 1, 28, 28)
    >>> out = model(x)
    >>> out.shape
    torch.Size([8, 10])
    """

    def __init__(
        self,
        num_filters: int = 16,
        hidden_size: int = 32,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters

        self.num_filters = num_filters
        self.hidden_size = hidden_size
        self.lr = lr

        self.conv = nn.Conv2d(1, self.num_filters, 3)
        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        # Calculate the flattened size after conv and pool
        self.fc1 = nn.Linear(self.num_filters * 13 * 13, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Output logits of shape (N, 10).
        """

        x = self.pool(F.relu(self.conv(x)))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Parameters
        ----------
        batch : tuple
            A batch of data (images, labels).
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Training loss.
        """

        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Parameters
        ----------
        batch : tuple
            A batch of data (images, labels).
        batch_idx : int
            Index of the batch.

        Returns
        -------
        None
        """

        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.

        Parameters
        ----------
        batch : tuple
            A batch of data (images, labels).
        batch_idx : int
            Index of the batch.

        Returns
        -------
        None
        """

        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """
        Define optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer with default parameters.
        """

        return optim.Adam(self.parameters(), lr=self.lr)
