import tensorflow as tf
import cv2
import numpy as np
import random
import threading
import queue
from tensorflow.keras.callbacks import EarlyStopping

# Data augmentation function
def data_augmentation(positive_sample, negative_sample):
    augmented_positive_samples = []
    augmented_negative_samples = []

    # Define transformations
    def random_rotation(image):
        angle = random.randint(-15, 15)
        return cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1), (image.shape[1], image.shape[0]))

    def random_translation(image):
        tx, ty = random.randint(-5, 5), random.randint(-5, 5)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    def random_brightness(image):
        alpha = 1.0 + 0.2 * random.uniform(-1, 1)
        beta = 30 * random.uniform(-1, 1)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    transformations = [random_rotation, random_translation, random_brightness]

    # Apply transformations
    for image in [positive_sample, negative_sample]:
        augmented_image = image.copy()
        for transform in transformations:
            if random.random() < 0.5:  # Apply each transformation with a 50% probability
                augmented_image = transform(augmented_image)
        if image is positive_sample:
            augmented_positive_samples.append(augmented_image)
        else:
            augmented_negative_samples.append(augmented_image)

    return augmented_positive_samples, augmented_negative_samples

# On-the-fly Trainer class

class OnTheFlyTrainer:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # Convolutional layer with 16 filters
            tf.keras.layers.Dropout(0.25),  # Dropout layer
            tf.keras.layers.MaxPooling2D(2, 2),  # MaxPooling layer
            tf.keras.layers.Flatten(),  # Flatten layer
            tf.keras.layers.Dense(64, activation='relu'),  # Dense layer with 64 nodes
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def train(self, positive_samples, negative_samples):
        X = np.concatenate([positive_samples, negative_samples])
        Y = np.concatenate([np.ones(len(positive_samples)), np.zeros(len(negative_samples))])

        # Split the data into training and validation
        split_idx = int(0.8 * len(X))
        X_train, Y_train = X[:split_idx], Y[:split_idx]
        X_val, Y_val = X[split_idx:], Y[split_idx:]

        # Early stopping callback
        early_stop = EarlyStopping(monitor='val_loss', patience=3)

        self.model.fit(X_train, Y_train, epochs=5, verbose=0, validation_data=(X_val, Y_val), callbacks=[early_stop])

    def predict(self, patch):
        return self.model.predict(np.array([patch]))

def train_worker(training_queue, trainer):
    while True:
        pos_samples, neg_samples = training_queue.get()
        if pos_samples is None:  # Sentinel value to exit thread
            break
        trainer.train(pos_samples, neg_samples)

# Initialize Trainer and Training Queue
trainer = OnTheFlyTrainer()
tracker = cv2.legacy.TrackerCSRT.create()
training_queue = queue.Queue()

# Start the training thread
training_thread = threading.Thread(target=train_worker, args=(training_queue, trainer))
training_thread.start()

# Kalman filter setup
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# OpenCV setup
cap = cv2.VideoCapture(0)
roi_selected = False
tracking = False
bbox = None
lk_params = dict(winSize=(30, 30), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = None
p0 = None

batch_positive_samples = []
batch_negative_samples = []
predict_frequency = 5
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if tracking:
        # Use CSRT tracker to track the object
        (success, newbox) = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in newbox]
            # Use deep learning model to verify
            patch = cv2.resize(frame[int(y):int(y + h), int(x):int(x + w)], (64, 64))
            prediction = trainer.predict(patch)
            if prediction > 0.5:
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            else:
                # Reinitialize the CSRT tracker
                tracker = cv2.TrackerCSRT_create()
                tracking = False  # Set tracking to False, so in the next loop, user can re-select ROI

        # Training the model for the current frame
        positive_sample = cv2.resize(frame[int(y):int(y + h), int(x):int(x + w)], (64, 64))
        x1, y1 = max(int(x - 10), 0), max(int(y - 10), 0)
        x2, y2 = min(int(x + w + 10), frame.shape[1]), min(int(y + h + 10), frame.shape[0])
        negative_sample_slice = frame[y1:y2, x1:x2]

        if negative_sample_slice.size > 0:
            negative_sample = cv2.resize(negative_sample_slice, (64, 64))
        else:
            continue
        augmented_positive_samples, augmented_negative_samples = data_augmentation(positive_sample, negative_sample)
        trainer.train(augmented_positive_samples, augmented_negative_samples)

    cv2.imshow("Tracking", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        if not roi_selected:
            bbox = cv2.selectROI(frame)
            if bbox:
                x, y, w, h = [int(v) for v in bbox]
                tracker.init(frame, bbox)
                roi_selected = True
                tracking = True  # Start tracking immediately after selecting the object

cap.release()
cv2.destroyAllWindows()
