# Image Rating Model

## Overview
This project implements an image rating system using a convolutional neural network (CNN). The model is based on the ResNet architecture modified for regression tasks. It is designed to rate images based on features extracted from them.

## Contents
- **/data**: Directory containing images and dataset files
- **/models**: Trained model weights
- **/src**: Source codes for dataset preparation, model definition, prediction, and training
- **/notebooks**: Jupyter notebooks for exploratory data analysis and visualization

## Requirements
- Python 3.x
- PyTorch
- TorchVision
- Pandas
- Pillow

### Installation
```bash
pip install -r requirements.txt
```

## Usage
**Data Preparation**:
- Place your images in the `/data/images` directory.
- Ensure the JSON containing ratings is in `/data/datasets`.
- Use the `train_test_splitter.py` script to split your dataset into training, validation, and holdout sets.
Run the following command in your terminal:
```bash 
python src/train_test_splitter.py\n     
```    
You can customize the ratios of the splits by modifying the ratios in the script if desired (default values are 0.8 for training and 0.15 for validation). Make sure to update the `json_file_path` variable in `train_test_splitter.py` to point to your JSON rating file and set the appropriate `output_dir` where you want the split datasets to be stored.

**Training the Model**:
- Run the following command:
```bash
python src/Trainer.py
``` 
**Predicting Ratings**: 
- Use the `Predictor.py` script to run predictions on new images.