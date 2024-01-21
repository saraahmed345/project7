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


# Set the root folder, target size, crop size, and standardization parameters
mFolder_path = r"C:\Users\DELL\Downloads\dataset NN'23 (1)\dataset\train"
cropping_size = (227, 227)  #224,224

# Create an ImageDataGenerator for data augmentation
img_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# Lists to store pixel values for mean and standard deviation calculation
pixels = []

# Lists to store (image, label) pairs
data = []

# Get the list of class folders
classes = [folder for folder in os.listdir(mFolder_path) if os.path.isdir(os.path.join(mFolder_path, folder))]

# Iterate through each class folder
for clas in classes:
    class_path = os.path.join(mFolder_path, clas)

    # Get the list of image files in the current class folder
    images = [file for file in os.listdir(class_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

    # Iterate through each image file in the current class folder
    for image in images:
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

        # Resize the image
        img = cv2.resize(img, (224, 224))

        # Convert to grayscale(optional)
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
        pixels.extend(cropped_image.flatten())

        # Convert the class folder name to an integer label
        label = int(clas)

        # Append the (image, label) pair to the dataset
        data.append((cropped_image, label))

# Calculate mean and standard deviation
pixels = np.array(pixels)
mean = np.mean(pixels)
std = np.std(pixels)

# Apply standardization to the entire dataset
for i in range(len(data)):
    data[i] = ((data[i][0] - mean) / std, data[i][1])

# Split the dataset into training and testing sets
X, y = zip(*data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


