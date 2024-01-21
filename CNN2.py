# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:50:31 2023

@author: DELL
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score


mFolder_path = r"C:\Users\DELL\Downloads\dataset NN'23 (1)\dataset\train"
cropping_size = (224, 224)  # Adjusted based on your cropping size

# Create an ImageDataGenerator for data augmentation
img_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# Lists to store pixel values for mean and standard deviation calculation
pixels = []

# Lists to store (image, image_id, label) triplets
data = []

# Define class names
class_names = ["Apples", "Banana", "Grapes", "Mango", "Strawberry"]

# Iterate through each class folder
for label, class_name in enumerate(class_names, start=1):
    class_path = os.path.join(mFolder_path, str(label))  # Adjusted to use numerical labels

    # Get the list of image files in the current class folder
    images = [file for file in os.listdir(class_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

    # Iterate through each image file in the current class folder
    for image_id, image in enumerate(images, start=1):
        image_path = os.path.join(class_path, image)

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file '{image_path}' not found. Skipping...")
            continue

        img = cv2.imread(image_path)

        # Check if the image is None (indicating a failure to read the image)
        if img is None:
            print(f"Warning: Unable to read image file '{image_path}'. Skipping...")
            continue

        # Resize the image to the desired dimensions
        img = cv2.resize(img, (128, 128))

        # Convert to grayscale (optional)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply data augmentation to the grayscale image
        img_gray = img_gray.reshape((1,) + img_gray.shape + (1,))  # Add channel dimension
        augmented_images = img_generator.flow(img_gray, batch_size=1)
        augmented_image = augmented_images[0][0]

        # Normalize pixel values to the range [0, 1]
        augmented_image = augmented_image / 255.0

        # Crop and center the image
        h, w = augmented_image.shape[:2]
        top = (h - cropping_size[0]) // 2
        left = (w - cropping_size[1]) // 2
        bottom = top + cropping_size[0]
        right = left + cropping_size[1]
        cropped_image = augmented_image[top:bottom, left:right]

        # Collect pixel values for mean and standard deviation calculation
        #cropped_image = cv2.resize(cropped_image, (new_width, new_height))
        #pixels.extend(resized_image.flatten())
        pixels.extend(cropped_image.flatten())

        # Append the (image, image_id, label) triplet to the dataset
        data.append((cropped_image, f"{class_name}_{image_id}", label))

# Calculate mean and standard deviation
pixels = np.array(pixels)
mean = np.mean(pixels)
std = np.std(pixels)

# Apply standardization to the entire dataset
for i in range(len(data)):
    data[i] = ((data[i][0] - mean) / std, data[i][1], data[i][2])

# Split the dataset into training and testing sets
X, image_ids, y = zip(*data)
X_train, X_test, image_ids_train, image_ids_test, y_train, y_test = train_test_split(X, image_ids, y, test_size=0.2, random_state=42)

# Print unique labels in the dataset
unique_labels = set(y)
print(f"Unique labels in the dataset: {unique_labels}")

# One-hot encode the labels
y_train_onehot = to_categorical(np.array(y_train) - 1, num_classes=len(class_names))
y_test_onehot = to_categorical(np.array(y_test) - 1, num_classes=len(class_names))


# Add channel dimension to X_train and X_test
X_train = np.array(X_train).reshape(-1, 48, 48, 1)
X_test = np.array(X_test).reshape(-1, 48, 48, 1)
# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))



# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape for training set
X_train = np.array([np.expand_dims(x, axis=-1) for x in X_train])  # Add channel dimension
y_train = np.array(y_train) - 1  

# Train the model
model.fit(X_train, y_train_onehot, epochs=10, batch_size=12, validation_split=0.2)



# Reshape for testing set
X_test = np.array([np.expand_dims(x, axis=-1) for x in X_test])  # Add channel dimension
y_test = np.array(y_test)-1




# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
# Set the path to the test folder
test_folder_path = r"C:\Users\DELL\Downloads\dataset NN'23 (1)\dataset\test"

# Lists to store (image_id, label) pairs for the test set
test_data = []

# Iterate through each image file in the test set
test_images = [file for file in os.listdir(test_folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

for image_id, image in enumerate(test_images, start=1):
    image_path = os.path.join(test_folder_path, image)

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Warning: Image file '{image_path}' not found. Skipping...")
        continue

    img = cv2.imread(image_path)

    # Check if the image is None (indicating a failure to read the image)
    if img is None:
        print(f"Warning: Unable to read image file '{image_path}'. Skipping...")
        continue

    # Resize the image to the desired dimensions
    img = cv2.resize(img, (48, 48))

    # Convert to grayscale (optional)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to the range [0, 1]
    img_gray = img_gray.reshape((1,) + img_gray.shape + (1,))  # Add channel dimension
    img_gray = img_gray / 255.0

    # Crop and center the image
    h, w = img_gray.shape[:2]
    top = (h - cropping_size[0]) // 2
    left = (w - cropping_size[1]) // 2
    bottom = top + cropping_size[0]
    right = left + cropping_size[1]
    cropped_image = img_gray[top:bottom, left:right]
    # Extract the label from the metadata file based on image ID
   

    
    test_data.append((int(image.split('.')[0]), cropped_image))  # Extract numeric part from filename

# Apply standardization to the entire test set
for i in range(len(test_data)):
    test_data[i] = (test_data[i][0], (test_data[i][1] - mean) / std)

# Extract features and labels for the test set
test_ids, test_images = zip(*test_data)
test_images = np.array(test_images).reshape(-1, 48, 48, 1)

# Make predictions on the test set
test_predictions_onehot = model.predict(test_images)

test_predictions_onehot
# Convert one-hot encoded predictions to class labels
test_predictions_labels = np.argmax(test_predictions_onehot, axis=1) + 1 

# Create a DataFrame for the test set predictions
test_results = pd.DataFrame({'image_id': test_ids, 'label': test_predictions_labels})

# Save the DataFrame to a CSV file with only 'image_id' and 'label'
csv_filename_test = r'C:\Users\DELL\Downloads\NeuralNetworkProject (2) (1)\NeuralNetworkProject/test_results.csv'
test_results[['image_id', 'label']].to_csv(csv_filename_test, index=False)

# Print the first few rows of the test results DataFrame
print("Test Results:")
print(test_results[['image_id', 'label']].head())

# Provide a link to download the simplified CSV file for the test set
print("Download the simplified CSV file for the test set:")
print(csv_filename_test)




