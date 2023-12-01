import cv2
import numpy as np
import os
import json
from tqdm import tqdm

# Directory containing frames
frames_dir = './triball_frames'
# Output JSON file with labels
output_json_path = 'train_disc_labels.json'

# Define the color range for green triballs
lower_green = np.array([36, 25, 25])  # Replace with your values
upper_green = np.array([86, 255, 255])  # Replace with your values

# Initialize a dictionary to hold label data
labels_dict = {}

# Get a sorted list of frame filenames
frame_files = sorted(os.listdir(frames_dir))
# Progress bar setup
pbar = tqdm(total=len(frame_files), unit="frame")

for frame_file in frame_files:
    # Progress bar update for each frame
    pbar.update(1)

    # Skip non-image files
    if not frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Read the image
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)

    # Check if the image was correctly loaded
    if frame is None:
        print(f"Could not read image {frame_file}.")
        continue

    # Convert the frame to the HSV color space and create a mask
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store bounding boxes for each triball in the frame
    triball_bboxes = []
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        triball_bboxes.append([int(x), int(y), int(x + w), int(y + h)])

    # Update the labels dictionary
    labels_dict[frame_file] = triball_bboxes

# Close the progress bar
pbar.close()

# Write the labels dictionary to a JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(labels_dict, json_file)

print(f"Labels saved to {output_json_path}")
