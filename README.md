# Detect-Pneumonia-Fall-2022-23---Kaggle
This Google Colab notebook implements a deep learning model for the Kaggle X-Ray Classification Challenge. The model is built using TensorFlow and uses transfer learning with the DenseNet121 architecture, pre-trained on the ImageNet dataset.
link to the contest: https://www.kaggle.com/competitions/detect-pneumonia-fall-2022-23

# Dataset
The dataset consists of X-Ray images of chests, and each image is labeled as normal, bacteria, or virus. The dataset is split into train and test datasets. The train dataset is further split into train and validation datasets.

#Getting Started
To get started with this notebook, follow these steps:

1. Download the train and test datasets from the link: https://www.kaggle.com/competitions/detect-pneumonia-fall-2022-23/data and upload them to your Google Drive.
2. Make sure you have a GPU runtime enabled in Google Colab.
3. Run the notebook cells in order.

# Libraries Used
This notebook uses the following libraries:

* google.colab
* pandas
* numpy
* matplotlib
* tensorflow
* keras
* Notebook Overview

# The notebook performs the following steps:

1. Mounts Google Drive
2. Checks if GPU is enabled
3. Loads the train labels from the CSV file
4. Sets the directories for train and test datasets
5. Loads the DenseNet121 model pre-trained on ImageNet and freezes its layers
6. Adds a Global Average Pooling layer and a Dense layer for classification
7. Compiles the model
8. Splits the train dataset into train and validation datasets using ImageDataGenerator
9. Trains the model using fit method and EarlyStopping callback
10. Plots the accuracy and loss curves
11. Loads the test dataset and predicts the class for each test image
12. Saves the predicted classes to a CSV file.
