from PIL import Image
import torch
from src.ImageRater import ImageRater
from torchvision import transforms
import os


def preprocess_image(image_path: str):
    """Preprocesses the image for model input.

    Args:
        image_path (str): Path to the image to be processed.

    Returns:
        Tensor: Preprocessed image tensor ready for model input.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to match model input size
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )
    image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(image_path: str, model_path: str):
    """Predicts the score for the given image using the trained model.

    Args:
        image_path (str): Path to the image for prediction.
        model_path (str): Path to the trained model's weights.

    Returns:
        float: Predicted score for the image.

    Raises:
        FileNotFoundError: If the specified image or model path does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ImageRater()
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.eval()
    image = preprocess_image(image_path)
    with torch.no_grad():
        score = model(image).item()
    return score


if __name__ == "__main__":
    # Example usage
    print(predict("new_image.jpg", "model_weights.pth"))
