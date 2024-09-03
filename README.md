# project7
Image Classification with Convolutional Neural Network
Overview
This repository contains a Python script for training and evaluating a Convolutional Neural Network (CNN) on an image classification task. The dataset consists of images of fruits, and the script includes functionality for data augmentation, preprocessing, model training, and testing.

Features
Data Augmentation: Utilizes Keras's ImageDataGenerator for augmenting training images.
Preprocessing: Includes image resizing, cropping, and normalization.
Model Architecture: A CNN model with two convolutional layers and max pooling.
Evaluation: Evaluates the model's performance on a test dataset and saves predictions to a CSV file.
Requirements
To run this script, you need the following Python packages:

tensorflow
keras
numpy
opencv-python
pandas
scikit-learn
You can install these packages using pip:

bash
Copy code
pip install tensorflow keras numpy opencv-python pandas scikit-learn
Directory Structure
Ensure your directory structure matches the following:

bash
Copy code
dataset/
    train/
        1/
            image1.jpg
            image2.jpg
            ...
        2/
        ...
    test/
        test_image1.jpg
        test_image2.jpg
        ...
train/: Contains subfolders named with numerical labels (e.g., 1, 2, etc.), each with images of a particular class.
test/: Contains images for which predictions will be made.
Script Overview
1. Data Preparation
Image Loading: Loads images from the train directory.
Preprocessing: Resizes, converts to grayscale, augments, crops, and normalizes images.
Dataset Splitting: Splits the data into training and testing sets.
2. Model Building
CNN Architecture: Defines a CNN with two convolutional layers, max pooling, and dense layers.
3. Training
Model Training: Trains the model using the training dataset with data augmentation.
4. Evaluation
Testing: Evaluates the model on the test dataset and saves the results to a CSV file.
5. Test Results
CSV Output: Saves predictions for the test images in test_results.csv.
Usage
Place Your Dataset: Ensure your dataset is organized as described in the directory structure.

Adjust Paths: Update mFolder_path and test_folder_path variables in the script to point to your dataset locations.

Run the Script: Execute the script using Python:

bash
Copy code
python CNN2.py
Check Results: After execution, check test_results.csv in the specified directory for the test results.

Example
Here is a sample output from the script:

yaml
Copy code
Unique labels in the dataset: {1, 2, 3, 4, 5}
Test Accuracy: 95.67%
Test Results:
   image_id  label
0         1      3
1         2      1
...


Notes
Ensure that your dataset is properly labeled and organized.
Adjust hyperparameters and model architecture as needed for your specific task.
