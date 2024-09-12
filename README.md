# Deepfake Detection

## Overview

This project aims to classify videos as "REAL" or "FAKE" using a deep learning model. The model architecture leverages the `InceptionV3` model for feature extraction, followed by `LSTM` and `GRU` layers for sequence processing.

[![Deepfake Detection Web APP](https://github.com/VMD7/blooms-taxonomy-classifier/blob/master/AnimationBlooms.gif)](https://vjdevane-deepfake-detection.hf.space)


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

## Dataset

The dataset used for training and evaluation consists of:
* 77 Fake Videos
* 76 Real Videos

Each video has 10 frames extracted based on motion detection.

This newly motion detected dataset is released on kaggel as [Deepfake Detection Challenge Dataset - Face Images](https://www.kaggle.com/datasets/vijaydevane/deepfake-detection-challenge-dataset-face-images).

## Training
The model was trained on [Kaggle Notebooks](https://www.kaggle.com/code/vijaydevane/deepfakedetectiontraining) with the following specifications:
* GPU Accelerators: T4 x 2
* Dataset: [Deepfake Detection Challenge Dataset](https://www.kaggle.com/c/deepfake-detection-challenge).

## Deployment

The whole project is deployed on the Hugging Face Hub. You can interact with the model it with this [link](https://vjdevane-deepfake-detection.hf.space).

## Contribution Guidelines
Contributions to the Deepfake Detection are welcome! If you have ideas for improvements, new features, or bug fixes, please initiate a thread.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact
For any inquiries or support regarding the Deepfake Detection, please contact [Vijay Devane](https://www.linkedin.com/in/vijay-devane-a629931b3/).
