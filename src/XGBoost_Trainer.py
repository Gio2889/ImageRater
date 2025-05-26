import os
import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from ImageRater import ImageRater
import torch
import torchvision.transforms as T
from PIL import Image  # Ensure to import PIL for image processing

class XGBoostModelTrainer:
    def __init__(self, model_type='resnet', transform = None,device : str = 'gpu'):
        """Initialize the XGBoost model trainer with the specified CNN model type."""
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- Device set to {self.device} ---")
        # Initialize the CNN model for feature extraction
        self.cnn_model = ImageRater(model_type=self.model_type).to(self.device)  # Adjust device as necessary
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
    
    def extract_features(self, image_paths):
        """Extract features from images using the CNN model."""
        features = []
        self.cnn_model.eval()
        with torch.no_grad():
            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # Add batch dimension
                feature = self.cnn_model(img_tensor)  # Get features from the CNN
                features.append(feature.cpu().numpy().flatten())  # # Move to CPU before converting to NumPy to Flatten and append
        return np.array(features)

    def feature_extractor(self, image_paths, ratings):
        """Extract features and return both feature vectors and ratings."""
        feature_vectors = self.extract_features(image_paths)
        rating_array = np.array(ratings)
        return feature_vectors, rating_array

    def train(self, X_train, y_train, X_test, y_test):
        """Train the XGBoost model using the provided training and testing datasets."""
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,  # Number of boosting rounds
            max_depth=5,        # Control model complexity
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            eval_metric='mae'
        )

        # Train with validation split
        model.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)],
                  verbose=True)

        # Predict and evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"MAE: {mae:.2f} (Target: ±1.0)")
    
        return model, mae

    @staticmethod
    def load_data(train_dir, test_dir):
        """Load the training and testing data from the specified directories."""
        def load_directory(directory):
            images = []
            ratings = []
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    json_file = os.path.join(directory, filename)
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        for image_name, rating in data.items():
                            img_path = os.path.join(directory, image_name)
                            images.append(img_path)
                            ratings.append(rating)
            return images, np.array(ratings, dtype=np.float32)  # Convert ratings to float32 for training

        X_train, y_train = load_directory(os.path.join(train_dir))
        X_test, y_test = load_directory(os.path.join(test_dir))
        return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    train_dir = "D:/DiscordBotTrainingSet/sets/train"
    test_dir = "D:/DiscordBotTrainingSet/sets/val"
    
    model_type = 'resnet'  # or 'efficientnet'
    trainer = XGBoostModelTrainer(model_type=model_type,device='cuda')
    
    X_train, y_train, X_test, y_test = trainer.load_data(train_dir, test_dir)
    
    feature_vectors_train, y_train = trainer.feature_extractor(X_train, y_train)
    feature_vectors_test, y_test = trainer.feature_extractor(X_test, y_test)

    trainer.train(feature_vectors_train, y_train, feature_vectors_test, y_test)