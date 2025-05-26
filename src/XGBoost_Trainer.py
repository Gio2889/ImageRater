import os
import json
import numpy as np
import xgboost as xgb
import hashlib
from sklearn.metrics import mean_absolute_error
from ImageRater import ImageRater
import torch
import torchvision.transforms as T
from PIL import Image  # Ensure to import PIL for image processing
from sklearn.model_selection import GridSearchCV


class XGBoostModelTrainer:
    def __init__(self, model_type="resnet", transform=None, device: str = "gpu"):
        """Initialize the XGBoost model trainer with the specified CNN model type."""
        self.model_type = model_type
        self.tune_parms = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Device set to {self.device} ---")
        # Initialize the CNN model for feature extraction
        self.cnn_model = ImageRater(model_type=self.model_type).to(
            self.device
        )  # Adjust device as necessary
        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

    def extract_features(self, image_paths):
        """Extract features from images using the CNN model."""
        features = []
        self.cnn_model.eval()
        with torch.no_grad():
            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB")
                img_tensor = (
                    self.transform(img).unsqueeze(0).to(self.device)
                )  # Add batch dimension
                feature = self.cnn_model(img_tensor)  # Get features from the CNN
                features.append(
                    feature.cpu().numpy().flatten()
                )  # # Move to CPU before converting to NumPy to Flatten and append
        return np.array(features)

    def feature_extractor(self, image_paths, ratings):
        """Extract features and return both feature vectors and ratings."""
        hash_key = self.generate_hash(
            os.path.dirname(image_paths[0])
        )  # Assuming all images are in the same directory
        cached_features = self.get_cached_features(hash_key, mode="train")

        if cached_features is not None:
            print(" -- Loaded cached features. --")
            return cached_features, np.array(ratings)

        feature_vectors = self.extract_features(image_paths)
        self.save_cached_features(feature_vectors, hash_key, mode="train")
        return feature_vectors, np.array(ratings)

    def train(self, X_train, y_train, X_test, y_test):
        """Train the XGBoost model using the provided training and testing datasets."""
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2000,  # Number of boosting rounds
            max_depth=10,  # Control model complexity
            learning_rate=0.001,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            eval_metric="mae",
        )

        # Train with validation split
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

        # Predict and evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"MAE: {mae:.2f} (Target: ±1.0)")

        return model, mae

    def generate_hash(self, directory):
        """Generate SHA-256 hash for the contents of the directory."""
        hasher = hashlib.sha256()
        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)
            if os.path.isdir(filepath):
                hasher.update(self.generate_hash(filepath).encode())
            else:
                hasher.update(filepath.encode())
        return hasher.hexdigest()

    def get_cached_features(self, hash_key, mode="train"):
        """Load cached features if they exist."""
        cache_path = (
            f"models/feature_vectors/train_fv_{hash_key}.npy"
            if mode == "train"
            else f"models/feature_vectors/test_fv_{hash_key}.npy"
        )
        if os.path.exists(cache_path):
            return np.load(cache_path)
        return None

    def save_cached_features(self, features, hash_key, mode="train"):
        """Save features to cache."""
        os.makedirs("models/feature_vectors", exist_ok=True)
        cache_path = (
            f"models/feature_vectors/train_fv_{hash_key}.npy"
            if mode == "train"
            else f"models/feature_vectors/test_fv_{hash_key}.npy"
        )
        np.save(cache_path, features)

    @staticmethod
    def load_data(train_dir, test_dir):
        """Load the training and testing data from the specified directories."""

        def load_directory(directory):
            images = []
            ratings = []
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    json_file = os.path.join(directory, filename)
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        for image_name, rating in data.items():
                            img_path = os.path.join(directory, image_name)
                            images.append(img_path)
                            ratings.append(rating)
            return images, np.array(
                ratings, dtype=np.float32
            )  # Convert ratings to float32 for training

        X_train, y_train = load_directory(os.path.join(train_dir))
        X_test, y_test = load_directory(os.path.join(test_dir))
        return X_train, y_train, X_test, y_test

    def tune_hyperparameters(self, X_train, y_train, X_test, y_test, param_grid=None):
        """Tune hyperparameters using GridSearchCV."""
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            }

        model = xgb.XGBRegressor(objective="reg:squarederror")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=5,  #
            verbose=1,
            n_jobs=-1,
        )  # Use all available cores

        grid_search.fit(X_train, y_train)
        print("--- Best parameters found: --- ")
        for key, val in grid_search.best_params_.items():
            print(f" -- {key}-->{val} --")

        # Extra training with the best parameters
        model.set_params(**grid_search.best_params_)
        model.fit(X_train, y_train)
        final_preds = model.predict(X_test)  # Predict on the training set
        final_mae = mean_absolute_error(y_test, final_preds)

        return model, final_mae

    def run_training_pipeline(
        self, train_dir, test_dir, tune_params=False, param_grid=None
    ):
        """Run the complete training pipeline including loading data, extracting features, and training the model."""
        print("--- creating data sets ---")
        X_train, y_train, X_test, y_test = self.load_data(train_dir, test_dir)
        print("--- data sets created ---\n")

        print("--- creating feature vectors ---")
        feature_vectors_train, y_train = self.feature_extractor(X_train, y_train)
        feature_vectors_test, y_test = self.feature_extractor(X_test, y_test)
        print("--- feature vectors created ---\n")

        if tune_params:
            print("--- Tuning hyperparameters ---")
            model, mae = self.tune_hyperparameters(
                feature_vectors_train, y_train, feature_vectors_test, y_test, param_grid
            )
        else:
            print("--- Training ---")
            model, mae = self.train(
                feature_vectors_train, y_train, feature_vectors_test, y_test
            )

        print(f"--- finished training with MAE: {mae:.2f} ---")
        return model, mae


if __name__ == "__main__":
    train_dir = "D:/DiscordBotTrainingSet/sets/train"
    test_dir = "D:/DiscordBotTrainingSet/sets/val"

    model_type = 'efficientnet'  # or 'efficientnet' "resnet"
    print("--- Initializing trainer ---")
    trainer = XGBoostModelTrainer(model_type=model_type, device="cuda")
    print("--- trainer ready ---\n")
    custom_training_grid = {
    "n_estimators": [50, 100, 200, 500, 1000],
    "max_depth": [3, 6, 10, 15],
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.5],
    "subsample": [0.2, 0.4, 0.6, 0.8, 1.0],
    "colsample_bytree": [0.2, 0.4, 0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 0.5, 1],
    "reg_lambda": [1, 1.5, 2],
    "scale_pos_weight": [1, 2, 3, 4],
    "objective": ["reg:squarederror", "binary:logistic"]
    }   

    # Run the training pipeline with hyperparameter tuning
    trainer.run_training_pipeline(
        train_dir, test_dir, tune_params=True, param_grid=None
    )
