# Deepfake Detection

## Overview

This project aims to classify videos as "REAL" or "FAKE" using a deep learning model. The model architecture leverages the `InceptionV3` model for feature extraction, followed by `LSTM` and `GRU` layers for sequence processing.

[![Deepfake Detection Web APP](https://github.com/VMD7/deepfake-detection/blob/main/Docs/AnimationDF.gif)](https://vjdevane-deepfake-detection.hf.space)

## Getting Started

### Clone the Repository:

```bash
git clone https://github.com/your-repo-link.git
cd your-repo-directory
```
### Install Dependencies:
```python
pip install -r requirements.txt
```

### Run the Application:
```python
python app.py
```

## Dataset

The dataset used for training and evaluation consists of:
* 77 Fake Videos
* 76 Real Videos

Each video has 10 frames extracted based on motion detection.

This newly motion detected dataset is released on kaggel as [Deepfake Detection Challenge Dataset - Face Images](https://www.kaggle.com/datasets/vijaydevane/deepfake-detection-challenge-dataset-face-images).


## Architecture

The model architecture consists of the following components:

1. **Feature Extraction**: 
   - `InceptionV3` pre-trained on ImageNet is used for feature extraction.
   - `GlobalAveragePooling2D` is applied to pool features from each frame.

2. **Sequential Processing**:
   - `TimeDistributed` applies the feature extractor to each frame independently.
   - `LSTM` and `GRU` layers are used to capture temporal dependencies.
   - `Dropout` is applied for regularization to prevent overfitting.
   - `Dense` layers are used for the final classification.

## Training
The model was trained on [Kaggle Notebooks](https://www.kaggle.com/code/vijaydevane/deepfakedetectiontraining) with the following specifications:
* GPU Accelerators: T4 x 2
* Dataset: [Deepfake Detection Challenge Dataset](https://www.kaggle.com/c/deepfake-detection-challenge).


## Demo

You can try out the model live on Hugging Face Spaces: [Demo](https://vjdevane-deepfake-detection.hf.space)

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes. Ensure your code adheres to the existing style and includes appropriate tests.
   
## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
