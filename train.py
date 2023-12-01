import tensorflow as tf
from tensorflow.keras import layers, models
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os


def load_and_preprocess_image(img_path, image_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0  # Normalize to [0, 1]
    return img


def extract_frames(video_path, output_dir):
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames and saved to directory: {output_dir}")


# Rest of the script...

# Extract frames from video
video_path = './m.mp4'
image_dir = './spinup'
extract_frames(video_path, image_dir)

# Load the labeled data
with open('train_disc_labels.json', 'r') as f:
    labels = json.load(f)


# Split filenames into training and validation sets
file_names = list(labels.keys())
train_files, val_files = train_test_split(file_names, test_size=0.1, random_state=42)

# Data generator
def data_generator(file_list, labels_dict, batch_size):
    while True:
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i+batch_size]
            images = []
            bboxes = []
            for file_name in batch_files:
                img_path = os.path.join(image_dir, file_name)
                img = load_and_preprocess_image(img_path, image_size)
                images.append(img)

                # Process bounding boxes
                normalized_bboxes = [coord / image_size for bbox in labels_dict[file_name] for coord in bbox]
                label = normalized_bboxes + [0] * (label_length - len(normalized_bboxes))
                label = label[:label_length]
                bboxes.append(label)

            yield np.array(images), np.array(bboxes)

# Parameters
max_triballs = 50  # Reduced number of maximum triballs
label_length = max_triballs * 4
image_size = 128  # Reduced image size

# Simplified Model architecture
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(label_length, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')


# Create data generators
batch_size = 16
train_generator = data_generator(train_files, labels, batch_size)
val_generator = data_generator(val_files, labels, batch_size)

# Train the model
model.fit(train_generator, validation_data=val_generator,
          steps_per_epoch=len(train_files) // batch_size,
          validation_steps=len(val_files) // batch_size,
          epochs=100)

# Save the model
model.save('robot_detection.h5')
