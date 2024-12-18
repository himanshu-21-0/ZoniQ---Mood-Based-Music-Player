import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
data_dir = "Data/Images"
model_save_path = "Models/retrained_graph.pb"
labels_save_path = "Models/retrained_labels.txt"

# Create Models folder if it doesn't exist
if not os.path.exists("Models"):
    os.makedirs("Models")

# Data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # Normalize pixel values
    rotation_range=20,          # Random rotation
    width_shift_range=0.2,      # Random horizontal shift
    height_shift_range=0.2,     # Random vertical shift
    shear_range=0.2,            # Random shearing
    zoom_range=0.2,             # Random zoom
    horizontal_flip=True,       # Flip images horizontally
    validation_split=0.2        # 20% of data for validation
)

# Training and validation datasets
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),       # Resize images to 48x48 pixels
    color_mode="grayscale",     # Grayscale images
    batch_size=32,
    class_mode="categorical",   # Multi-class classification
    subset="training"           # Training subset
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="validation"         # Validation subset
)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data
)

# Save the model
model.save(model_save_path)

# Save labels
class_labels = list(train_data.class_indices.keys())
with open(labels_save_path, "w") as f:
    f.write("\n".join(class_labels))

print("Model and labels saved successfully!")
