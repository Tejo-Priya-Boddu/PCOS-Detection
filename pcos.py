import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import tensorflow as tf

# Define the path to your dataset zip file
dataset_zip_path = './PCOS.zip'
dataset_extracted_path = './PCOS'  # Where to extract the dataset

# Step 1: Extract the dataset zip file
with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_extracted_path)

# Step 2: Use ImageDataGenerator to load images from the extracted folders
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=15,  # Reduced rotation range
    width_shift_range=0.15,  # Reduced width shift
    height_shift_range=0.15,  # Reduced height shift
    shear_range=0.15,  # Reduced shear range
    zoom_range=0.15,  # Reduced zoom range
    horizontal_flip=True,
    validation_split=0.05  # Use 5% of the data for validation
)

# Define the image size and batch size
img_size = (224, 224)  # Resize all images to this size for VGG16
batch_size = 32

# Load training images (95% training, 5% validation split)
train_generator = train_datagen.flow_from_directory(
    dataset_extracted_path,  # Directory with the dataset
    target_size=img_size,    # Resize images
    batch_size=batch_size,
    class_mode='binary',     # Binary classification
    subset='training'        # Set for training
)

validation_generator = train_datagen.flow_from_directory(
    dataset_extracted_path,  # Directory with the dataset
    target_size=img_size,    
    batch_size=batch_size,
    class_mode='binary',     
    subset='validation'      # Set for validation
)

# Step 3: Build a model using VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the base model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())  # Global average pooling layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))  # Increased dropout for regularization
model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.000005), loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model with early stopping and model checkpointing
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)  # Save as .keras
]

history = model.fit(train_generator,
                    epochs=10,  # Reduced number of epochs
                    validation_data=validation_generator,
                    callbacks=callbacks)

# Step 5: Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Validation Accuracy: {test_acc*100:.2f}%")

# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Step 6: Save the final model
model.save('pcos_detection_model.h5')  # Save the final model
