import torch
import os
import itertools
from torch.utils.data import DataLoader
from src.ImageRater import ImageRater
from src.dataset_creator import ImageDataset
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T


class ModelTrainer:
    """Initializes the Model Trainer with the necessary configurations.

    Args:
        config (dict): Configuration dictionary containing hyperparameters.
    """

    def __init__(self, main_path, config: dict):
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print("--- Initializing ModelTrainer ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImageRater().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        self.best_val_loss = float("inf")
        self.main_path = main_path
        self.train_transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_dataset = ImageDataset(
            "train_set.json",
            "f{self.main_path}sets/train/",
            transform=self.train_transform,
        )

        self.val_dataset = ImageDataset(
            "val_set.json", "f{self.main_path}/sets/val/", transform=self.val_transform
        )

        print(f"--- Model is on device: {next(self.model.parameters()).device} ---")

    def _validate_model(self):
        """Validates the model and returns the average validation loss.

        Returns:
            float: Average validation loss of the model.

        Raises:
            Exception: If loss computation fails during validation.
        """
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)
        val_loss = 0

        with torch.no_grad():
            for images, scores in val_loader:
                images, scores = images.to(self.device), scores.to(self.device)
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, scores)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"--- Validation Loss: {avg_val_loss:.4f} ---")
        return avg_val_loss

    def run(self):
        """Starts the training process for the model.

        Raises:
            Exception: If training encounters any unexpected issues during execution.
        """
        print("Starting training...")
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=64,
                shuffle=True,
                num_workers=4,
                pin_memory=False,
            )
            for images, scores in train_loader:
                images, scores = images.to(self.device), scores.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, scores)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

            val_loss = self._validate_model()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                os.makedirs("weights", exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    f"models/best_model_epoch_{self.best_epoch}.pth",
                )


def hyper_param_tuner(main_path: str, param_grid: dict = None):
    """Tunes hyperparameters using grid search.

    Args:
        param_grid (dict, optional): Dictionary containing parameters for grid search.

    Returns:
        None
    """
    if param_grid is None:
        print("-- No parameter grid provided --")
        param_grid = {
            "learning_rate": [1e-3],
            "batch_size": [64],
            "epochs": [20],
            "dropout_rate": [0.3],
        }
        print("Using default single value grid")
    grid = list(
        itertools.product(
            param_grid["learning_rate"],
            param_grid["batch_size"],
            param_grid["epochs"],
            param_grid["dropout_rate"],
        )
    )

    best_loss = float("inf")
    best_params = None

    for params in grid:
        config = {
            "learning_rate": params[0],
            "batch_size": params[1],
            "num_epochs": params[2],
            "dropout_rate": params[3],
        }

        trainer = ModelTrainer(main_path, config)
        trainer.run()
        if trainer.best_val_loss < best_loss:
            best_loss = trainer.best_val_loss
            best_params = config

    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_loss}")


if __name__ == "__main__":
    param_grid = {
        "learning_rate": [1e-3, 1e-4],
        "batch_size": [16, 32, 64],
        "epochs": [10, 20, 50],
        "dropout_rate": [0.3, 0.5],
    }
    main_path = "D:/DiscordBotTrainingSet/"
    print("--- Hypertuner initialized ---")
    hyper_param_tuner(main_path, param_grid)
    print("--- Hypertuner finished ---")
