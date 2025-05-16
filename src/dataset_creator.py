import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    """Initializes the ImageDataset class.

    Args:
        file_path (str): Path to the JSON file containing image paths and scores.
        prefix_path (str): Base path to prepend to image paths.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, file_path, prefix_path, transform=None):
        self.prefix_path = prefix_path
        self.df = pd.read_json(f"{prefix_path}{file_path}", typ="series")
        self.df = self.df.reset_index()
        self.df.columns = ["image_path", "score"]
        self.transform = transform

    def __len__(self):
        """Returns the number of items in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Fetches the image and score for the given index.
        Args:
            idx (int): Index of the item to fetch.

        Returns:
            tuple: (image, score) where image is the transformed image tensor and
            score is the corresponding score tensor.

        Raises:
            FileNotFoundError: If the image file at the specified path does not exist.
        """
        img_path = f"{self.prefix_path}/{self.df.iloc[idx]['image_path']}"
        score = self.df.iloc[idx]["score"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error in image on path\n {img_path}\n")
            print(f"Error: {e}")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(score, dtype=torch.float32)
