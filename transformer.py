import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tensorflow as tf
import pandas as pd

# Set the root folder, target size, crop size, and standardization parameters
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

# Define the Vision Transformer model from scratch
def create_vit_model(image_size, num_classes, patch_size, num_patches, embedding_dim, num_heads, mlp_dim, dropout_rate):
    inputs = layers.Input(shape=image_size)

    # Patching layer
    patching = layers.Conv2D(embedding_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    reshaped_patches = layers.Reshape((num_patches, embedding_dim))(patching)

    # Positional encoding
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)(tf.range(num_patches))
    positional_encoded_patches = reshaped_patches + position_embedding

    # Transformer Encoder
    for _ in range(num_heads):
        attention_output = layers.MultiHeadAttention(num_heads=1, key_dim=embedding_dim)(positional_encoded_patches, positional_encoded_patches)
        residual1 = layers.Add()([attention_output, positional_encoded_patches])
        normalized1 = layers.LayerNormalization(epsilon=1e-6)(residual1)

        mlp_output = layers.Dense(mlp_dim, activation="relu")(normalized1)
        mlp_output = layers.Dropout(dropout_rate)(mlp_output)
        mlp_output = layers.Dense(embedding_dim)(mlp_output)

        residual2 = layers.Add()([mlp_output, normalized1])
        normalized2 = layers.LayerNormalization(epsilon=1e-6)(residual2)

        positional_encoded_patches = normalized2

    # Global Average Pooling
    gap = layers.GlobalAveragePooling1D()(positional_encoded_patches)

    # Classifier
    outputs = layers.Dense(num_classes, activation="softmax")(gap)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Define model parameters
image_size = (48, 48, 1)  # Adjusted based on your preprocessing
num_classes = len(class_names)
patch_size = 16
num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
embedding_dim = 256
num_heads = 4
mlp_dim = 512
dropout_rate = 0.1

# Create the Vision Transformer model
vit_model = create_vit_model((48, 48, 1), num_classes, patch_size, num_patches, embedding_dim, num_heads, mlp_dim, dropout_rate)

# Compile the model with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
vit_model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # Changed to categorical_crossentropy for one-hot encoded labels
    metrics=['accuracy']
)

# Add channel dimension to X_train and X_test
X_train = np.array(X_train).reshape(-1, 48, 48, 1)
X_test = np.array(X_test).reshape(-1, 48, 48, 1)

# Print unique labels in the dataset
unique_labels = set(y)
print(f"Unique labels in the dataset: {unique_labels}")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ... (your existing code)

# Train the model
vit_model.fit(X_train, y_train_onehot, epochs=20, batch_size=12, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = vit_model.evaluate(X_test, y_test_onehot)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Make predictions on the test set
y_pred_onehot = vit_model.predict(X_test)

# Convert one-hot encoded predictions to class labels
y_pred_labels = np.argmax(y_pred_onehot, axis=1) + 1  # Adding 1 to convert back to your original class labels

# Convert one-hot encoded true labels to class labels
y_true_labels = np.argmax(y_test_onehot, axis=1) + 1  # Adding 1 to convert back to your original class labels

# Create confusion matrix
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))
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

    # Collect (image_id, label) pairs for the test set
    test_data.append((int(image.split('.')[0]), cropped_image))  # Extract numeric part from filename

# Apply standardization to the entire test set
for i in range(len(test_data)):
    test_data[i] = (test_data[i][0], (test_data[i][1] - mean) / std)

# Extract features and labels for the test set
test_ids, test_images = zip(*test_data)
test_images = np.array(test_images).reshape(-1, 48, 48, 1)

# Make predictions on the test set
test_predictions_onehot = vit_model.predict(test_images)

# Convert one-hot encoded predictions to class labels
test_predictions_labels = np.argmax(test_predictions_onehot, axis=1) + 1  # Adding 1 to convert back to your original class labels

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