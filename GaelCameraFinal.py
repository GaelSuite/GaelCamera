import cv2
import time
import onnxruntime as ort
import tensorflow as tf
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from sklearn.externals.joblib import Memory
except:
    from joblib import Memory
from sklearn.compose import ColumnTransformer
from FairMOT.src.lib.tracker.multitracker import JDETracker
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter

# Setting up the cache directory
memory = Memory(cachedir='/var/spinupcache/', verbose=0)

# Global variables
roi_selected = False
roi_being_selected = False
tracker_initialized = False
top_left_pt, bottom_right_pt = [], []

heatmap = None

past_points = []
degree = 2
max_past_points = 10
prediction_length = 3
speed_threshold = 5

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

selected_points = []
field_dims_in_feet = (12, 12)  # Vex field dimensions in feet

# Load the disc detection model
disc_detection_model = tf.keras.models.load_model('robot_detection.h5')
tracker = JDETracker(cfg, args.track_thresh)

def calculate_acceleration(current_speed, last_speed, time_interval):
    return (current_speed - last_speed) / time_interval

# def segment_color_objects(frame, lower, upper, field_mask):
#    """Segment objects of a specific color range inside the field and return their bounding boxes."""
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    mask = cv2.inRange(hsv, lower, upper)
#
#    # Mask out areas outside the field
#    mask &= field_mask
#
#    mask = cv2.erode(mask, None, iterations=2)
#    mask = cv2.dilate(mask, None, iterations=2)
#    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#    bounding_boxes = []
#    for contour in contours:
#        # Filter based on aspect ratio, extent, and size
#        x, y, w, h = cv2.boundingRect(contour)
#        aspect_ratio = float(w) / h
#        extent = cv2.contourArea(contour) / (w * h)
#
#        min_size = 20  # Adjust this based on the smallest game object
#        max_size = 1000  # Adjust this based on the largest game object
#
#        if 0.1 <= aspect_ratio <= 2.5 and 0.2 < extent and min_size < w * h < max_size:
#            bounding_boxes.append((x, y, w, h))
#    return bounding_boxes

def prepare_data_for_timeseries(points, times):
    X = np.array(times).reshape(-1, 1)  # Reshape times to be a 2D array
    y = np.array(points)  # Points are already in the correct format
    return X, y


def predict_future_positions(X, y, future_times, degree=3, uncertainty_factor=0.1, mc_simulations=1000):
    def create_polynomial_model(degree):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression())

    def calculate_rmse(actual, predicted):
        mse = mean_squared_error(actual, predicted)
        return np.sqrt(mse)

    def add_noise(data, noise_level):
        return data + np.random.normal(0, noise_level, data.shape)

    def run_monte_carlo_simulations(X, y, model, simulations, noise_level):
        mc_predictions = np.zeros((simulations, len(future_times), y.shape[1]))
        for i in range(simulations):
            noisy_y = add_noise(y, noise_level)
            model.fit(X, noisy_y)
            mc_predictions[i] = model.predict(np.array(future_times).reshape(-1, 1))
        return mc_predictions

    def calculate_confidence_intervals(mc_predictions, confidence_level=95):
        lower_percentile = (100 - confidence_level) / 2
        upper_percentile = 100 - lower_percentile
        lower_bounds = np.percentile(mc_predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(mc_predictions, upper_percentile, axis=0)
        return lower_bounds, upper_bounds

    def apply_kalman_filter(y):
        kf = KalmanFilter(initial_state_mean=y[0], n_dim_obs=2)
        kf = kf.em(y, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(y)
        return smoothed_state_means

    def generate_detailed_output(predictions, lower_bounds, upper_bounds):
        detailed_output = []
        for idx, time in enumerate(future_times):
            pred_x, pred_y = predictions[idx]
            lower_x, lower_y = lower_bounds[idx]
            upper_x, upper_y = upper_bounds[idx]
            detailed_output.append({
                'time': time,
                'predicted_position': (pred_x, pred_y),
                'confidence_interval': {
                    'x': (lower_x, upper_x),
                    'y': (lower_y, upper_y)
                }
            })
        return detailed_output

    # Apply Kalman Filter to smooth the input data
    smoothed_y = apply_kalman_filter(y)

    # Polynomial Regression Model and Predictions
    poly_model = create_polynomial_model(degree)
    poly_model.fit(X, smoothed_y)
    predictions = poly_model.predict(np.array(future_times).reshape(-1, 1))

    # Error Estimation and Monte Carlo Simulations
    y_pred = poly_model.predict(X)
    rmse = calculate_rmse(smoothed_y, y_pred)
    mc_predictions = run_monte_carlo_simulations(X, smoothed_y, poly_model, mc_simulations, rmse * uncertainty_factor)

    # Confidence Intervals
    lower_bounds, upper_bounds = calculate_confidence_intervals(mc_predictions)
    detailed_output = generate_detailed_output(predictions, lower_bounds, upper_bounds)
    return detailed_output

def segment_color_objects(frame, lower, upper, field_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask &= field_mask
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(mask, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(frame, markers)
    mask[markers == -1] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    def enhance_mask(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
        return mask

    def extract_features(contour, frame):
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        texture = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        mean_color = cv2.mean(roi)[:3]
        return {'histogram': hist, 'texture': texture, 'mean_color': mean_color}

    def filter_contours_by_size(contours, min_size=100, max_size=1000):
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size < area < max_size:
                filtered_contours.append(contour)
        return filtered_contours

    mask = enhance_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours_by_size(contours)

    detailed_objects = []
    for contour in contours:
        features = extract_features(contour, frame)
        x, y, w, h = cv2.boundingRect(contour)
        detailed_objects.append({'bbox': (x, y, w, h), 'features': features})

    return detailed_objects

# def segment_color_objects(frame, lower, upper, field_mask):
#     # Convert to the chosen color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Noise reduction
#     hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
#
#     # Color thresholding
#     mask = cv2.inRange(hsv, lower, upper)
#
#     # Morphological operations to clean up the mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#
#     # Apply the field mask to focus only on the relevant area
#     mask &= field_mask
#
#     # Find contours
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Filter contours by area
#     filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1 and cv2.contourArea(cnt) < 1700]
#
#     # Calculate bounding boxes from the filtered contours
#     bounding_boxes = [cv2.boundingRect(cnt) for cnt in filtered_contours]
#
#     return bounding_boxes

# Function to load the super-resolution model
def load_super_resolution_model(model_path):
    ort_session = ort.InferenceSession(model_path)
    return ort_session

# Function to apply super resolution to an image frame
def apply_super_resolution(frame, ort_session):
    # Preprocess the frame for the model (resize, normalize, etc.)
    # Convert to YCbCr color space and work on the Y channel (luminance)
    img_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    img_y = img_ycbcr[:, :, 0]
    img_y = img_y.astype(np.float32) / 255.
    img_y = np.expand_dims(np.expand_dims(img_y, 0), 0)

    # Run the model
    outputs = ort_session.run(None, {'input': img_y})
    enhanced_img_y = outputs[0]

    # Postprocess the output and convert back to BGR color space
    enhanced_img_y = np.squeeze(enhanced_img_y)
    enhanced_img_y = (enhanced_img_y * 255).clip(0, 255).astype(np.uint8)
    img_ycbcr[:, :, 0] = enhanced_img_y
    enhanced_frame = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2BGR)

    return enhanced_frame

def display_ar_overlays(frame, center, speed, acceleration, total_distance, robot_id="R001"):
    text_color = (0, 255, 255)  # Yellow
    cv2.putText(frame, f"Speed: {speed:.2f} ft/s", (center[0] + 20, center[1] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"Acceleration: {acceleration:.2f} ft/s^2", (center[0] + 20, center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"Total Distance: {total_distance:.2f} ft", (center[0] + 20, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"ID: {robot_id}", (center[0] + 20, center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

# Numeric feature processing
numeric_features = ['disc', 'robot', 'goal']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2)),
    ('feature_selection', SelectKBest(f_classif, k=10))
])

# Categorical feature processing
categorical_features = ['direction', 'off/def']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Feature union
feature_union = FeatureUnion([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=5))
])

def detect_discs(frame, model, confidence_threshold=0.5):
    # Adjust these parameters based on your model's input requirements
    input_size = (224, 224)  # Example size, change as needed

    # Preprocess the frame
    resized_frame = cv2.resize(frame, input_size)
    normalized_frame = resized_frame / 255.0
    processed_frame = np.expand_dims(normalized_frame, axis=0)

    # Run the model
    predictions = model.predict(processed_frame)[0]  # Assuming the model outputs an array

    # Postprocess the predictions
    detected_objects = []
    for pred in predictions:
        # Extract bounding box coordinates and confidence
        x, y, w, h, confidence = pred[:5]
        if confidence < confidence_threshold:
            continue

        # Convert coordinates from relative to absolute
        x, y, w, h = [int(dim) for dim in [x * frame.shape[1], y * frame.shape[0],
                                           w * frame.shape[1], h * frame.shape[0]]]

        detected_objects.append((x, y, w, h, confidence))

    # Apply Non-Maximum Suppression (NMS)
    boxes = np.array([obj[:4] for obj in detected_objects])
    confidences = np.array([obj[4] for obj in detected_objects])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)  # 0.4 is the NMS threshold

    # Filter out objects after NMS
    nms_filtered_objects = [detected_objects[i[0]] for i in indices]

    return nms_filtered_objects

# Ensemble classifiers
ensemble_classifier = FeatureUnion([
    ('svm', SVC(probability=True)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier())
])

pipeline_steps = [
    ('feature_processing', feature_union),
    ('classifier', ensemble_classifier)
]

def select_roi(event, x, y, flags, param):
    global top_left_pt, bottom_right_pt, roi_selected, roi_being_selected, x_cur, y_cur
    x_cur, y_cur = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        if not roi_being_selected:
            roi_selected = False
            top_left_pt = [(x, y)]
            roi_being_selected = True
    elif event == cv2.EVENT_LBUTTONUP:
        bottom_right_pt = [(x, y)]
        roi_selected = True
        roi_being_selected = False
        cv2.rectangle(frame, top_left_pt[0], bottom_right_pt[0], (0, 255, 0), 2)
        cv2.imshow('Tracking', frame)


def get_field_corners(event, x, y, flags, param):
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)  # Draw red dot
        if len(selected_points) > 1:
            cv2.line(temp_frame, selected_points[-2], selected_points[-1], (255, 0, 0), 2)  # Draw blue line
        cv2.imshow('Select Field Corners', temp_frame)
        if len(selected_points) == 4:
            cv2.line(temp_frame, selected_points[-1], selected_points[0], (255, 0, 0), 2)  # Close the polygon
            cv2.imshow('Select Field Corners', temp_frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

def warp_coordinates(point, matrix): # as u can see i had to do alot of debugging with this function with initial development
    if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
        raise ValueError("Invalid transformation matrix. It should be a 3x3 NumPy array.")

    if not isinstance(point, tuple) or len(point) != 2:
        raise ValueError("Invalid input point. It should be a tuple of (x, y) coordinates.")

    homogeneous_point = np.array([point[0], point[1], 1])

    try:
        # Perform the perspective transformation
        warped_point = np.dot(matrix, homogeneous_point)
    except Exception as e:
        raise RuntimeError(f"Error occurred during perspective transformation: {str(e)}")

    if warped_point[2] == 0:
        raise ValueError("Transformation matrix resulted in division by zero. Check matrix validity.")

    # Calculate the transformed (x, y) coordinates
    transformed_x = warped_point[0] / warped_point[2]
    transformed_y = warped_point[1] / warped_point[2]

    # Ensure the transformed coordinates are within reasonable bounds
    if not (0 <= transformed_x <= 1) or not (0 <= transformed_y <= 1):
        raise ValueError("Transformed coordinates are out of bounds. Check transformation matrix.")

    return (transformed_x, transformed_y)

def calculate_speed(point1, point2, time_interval=1.0):
    if not isinstance(point1, tuple) or not isinstance(point2, tuple):
        raise ValueError("Input points should be tuples of (x, y) coordinates.")

    if len(point1) != 2 or len(point2) != 2:
        raise ValueError("Input points should be tuples of (x, y) coordinates.")

    displacement = np.array(point2) - np.array(point1)
    speed = np.linalg.norm(displacement)
    velocity_x = displacement[0] / time_interval
    velocity_y = displacement[1] / time_interval

    result = {
        'speed': speed,
        'velocity_x': velocity_x,
        'velocity_y': velocity_y,
        'time_duration': time_interval
    }

    return result


def predict_trajectory(points, degree=2, length=3):
    if len(points) <= degree:
        return [points[-1]] * length

    # Extract x and y coordinates from points
    x = np.array(range(len(points)))
    y = np.array([p[0] for p in points])
    z = np.array([p[1] for p in points])

    # Perform polynomial regression for x and y coordinates
    coefficients_x = np.polyfit(x, y, degree)
    coefficients_y = np.polyfit(x, z, degree)

    # Generate future positions using polynomial extrapolation
    future_positions_x = [np.polyval(coefficients_x, len(points) + i) for i in range(length)]
    future_positions_y = [np.polyval(coefficients_y, len(points) + i) for i in range(length)]

    # Handling edge cases for out-of-bounds predictions
    for i in range(length):
        if future_positions_x[i] < min(x):
            future_positions_x[i] = min(x)
        elif future_positions_x[i] > max(x):
            future_positions_x[i] = max(x)

        if future_positions_y[i] < min(z):
            future_positions_y[i] = min(z)
        elif future_positions_y[i] > max(z):
            future_positions_y[i] = max(z)

    # Return a list of predicted future positions
    return list(zip(future_positions_x, future_positions_y))


def get_speed_color(start_point, end_point):
    dist = np.linalg.norm(np.array(start_point) - np.array(end_point))
    max_speed = 100
    ratio = np.clip(dist / max_speed, 0, 1)
    r = int(255 * ratio)
    g = int(255 * (1 - abs(2 * ratio - 1)))
    b = int(255 * (1 - ratio))
    return (b, g, r)

def get_tracking_box_color(last_speed, current_speed):
    speed_change = abs(current_speed - last_speed)
    if speed_change < 5:
        return (0, 255, 0)
    elif 5 <= speed_change < 15:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


cap = cv2.VideoCapture('m.mp4')
cv2.namedWindow('Select Field Corners')
cv2.setMouseCallback('Select Field Corners', get_field_corners)
ret, temp_frame = cap.read()
frame_copy = temp_frame.copy()

while len(selected_points) < 4:
    cv2.imshow('Select Field Corners', temp_frame)
    cv2.waitKey(1)

dst_points = np.array([
    [0, 0],
    [field_dims_in_feet[0], 0],
    [field_dims_in_feet[0], field_dims_in_feet[1]],
    [0, field_dims_in_feet[1]]
], dtype="float32")
matrix = cv2.getPerspectiveTransform(np.array(selected_points, dtype="float32"), dst_points)


def apply_field_overlay(frame, mask, color=(0, 255, 255)):
    """Apply an overlay to the field area."""
    overlay = frame.copy()
    overlay[mask == 255] = color
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


# After getting the field's boundaries using the selected points, create a mask
field_mask = np.zeros_like(temp_frame[:, :, 0], dtype=np.uint8)
field_contour = np.array(selected_points).reshape((-1, 1, 2)).astype(np.int32)
cv2.drawContours(field_mask, [field_contour], -1, (255), thickness=cv2.FILLED)

# Now, for the main tracking loop
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_roi)
tracker = cv2.TrackerCSRT_create()
fps = cap.get(cv2.CAP_PROP_FPS)
delay = 1.0 / fps
frames_to_skip = 0
last_speed = 0.0
total_distance = 0.0
start_time_for_roi_selection = None
roi_selection_time_limit = 3  # 3 seconds for ROI selection

data_time_x = []
data_time_y = []

max_skip_frames = 5  # Maximum number of frames to skip
min_skip_frames = 1  # Minimum number of frames to skip
skip_frame_threshold = 15  # Speed threshold to reduce frame skipping

ort_session = load_super_resolution_model('super_resolution.onnx')
pipeline = Pipeline(pipeline_steps, memory=memory)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    frame = cv2.warpPerspective(frame, matrix, (field_dims_in_feet[0], field_dims_in_feet[1]))
    frame = apply_super_resolution(frame, ort_session)
    frame_ = cv2.UMat(frame)

    past_points = []
    timestamps = []

    # Start the timer once the main loop begins
    if start_time_for_roi_selection is None:
        start_time_for_roi_selection = start_time

    # Calculate elapsed time
    elapsed_time = start_time - start_time_for_roi_selection
    yellow_bounding_boxes = segment_color_objects(frame, lower_yellow, upper_yellow, field_mask)

    # Check direction for color
    if len(past_points) > 2:
        delta_x = past_points[-1][0] - past_points[-2][0]
        box_color = (255, 0, 0) if delta_x > 0 else (0, 0, 255)
    else:
        box_color = (255, 0, 0)  # Default to red if we don't have enough points

    for (x, y, w, h) in yellow_bounding_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

    if not ret:
        break

    if frames_to_skip > 0:
        frames_to_skip -= 1
        continue

    if heatmap is None:
        heatmap = np.zeros_like(frame[:, :, 0]).astype(float)

    if roi_being_selected:
        cv2.rectangle(frame, top_left_pt[0], (x_cur, y_cur), (0, 255, 0), 2)

    detected_discs = detect_discs(frame, disc_detection_model, confidence_threshold=0.6)
    for (x, y, w, h, confidence) in detected_discs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if roi_being_selected:
        cv2.rectangle(frame, top_left_pt[0], (x_cur, y_cur), (0, 255, 0), 2)
        cv2.imshow('Tracking', frame)

    if roi_selected and not tracker_initialized:
        bbox = (top_left_pt[0][0], top_left_pt[0][1], bottom_right_pt[0][0] - top_left_pt[0][0], bottom_right_pt[0][1] - top_left_pt[0][1])
        # Initialize FairMOT with the selected region
        tracker.init(frame, bbox)
        tracker_initialized = True

    if tracker_initialized:
        ret, bbox = tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))


            heatmap[p1[1]:p2[1], p1[0]:p2[0]] += 1

            past_points.append(center)
            if len(past_points) > max_past_points:
                past_points.pop(0)

            center_real_world = warp_coordinates(center, matrix)
            if len(past_points) >= 2:
                speed_value = calculate_speed(warp_coordinates(past_points[-2], matrix), center_real_world) / delay
                acceleration = calculate_acceleration(speed_value, last_speed, delay)
                total_distance += speed_value * delay
                last_speed = speed_value
                display_ar_overlays(frame, center, speed_value, acceleration, total_distance)

                if speed_value > speed_threshold:
                    trajectory = predict_trajectory(past_points, degree, prediction_length)
                    arrow_color = get_speed_color(center, trajectory[-1])
                    for i in range(len(trajectory) - 1):
                        cv2.line(frame, tuple(map(int, trajectory[i])), tuple(map(int, trajectory[i + 1])), arrow_color,
                                 2)
                    cv2.arrowedLine(frame, tuple(map(int, trajectory[-2])), tuple(map(int, trajectory[-1])),
                                    arrow_color, 2, tipLength=0.5)

                box_color = get_tracking_box_color(
                    calculate_speed(past_points[-3], past_points[-2]) if len(past_points) >= 3 else 0, speed_value)
                cv2.rectangle(frame, p1, p2, box_color, 2)

            x, y, w, h = bbox
            center_x, center_y = x + w / 2, y + h / 2

            past_points.append((p1, p2))
            timestamps.append(time.time())

            future_times = [start_time + delta for delta in range(1, 6)]
            X, y = prepare_data_for_timeseries(past_points, timestamps)
            predicted_positions = predict_future_positions(X, y, future_times)

            direction = "static"
            if len(past_points) > 1:
                delta_x, delta_y = center_x - past_points[-1][0], center_y - past_points[-1][1]
                if abs(delta_x) > 0.3:  # 'threshold' is a small value to filter out noise
                    direction = "right" if delta_x > 0 else "left"
                if abs(delta_y) > 0.3:
                    direction = "down" if delta_y > 0 else "up"

                pov_angle = np.arctan2(center_y, center_x)
                mean_color = frame[y:y + h, x:x + w].mean(axis=(0, 1))

                data_time_x.append({
                    "center": (center_x, center_y),
                    "speed": speed_value,
                    "acceleration": acceleration,
                    "direction": direction,
                    "pov_angle": pov_angle
                })

                data_time_y.append({
                    "bbox": bbox,
                    "mean_color": mean_color,
                    "predicted_position": predicted_positions
                })

                pipeline.fit(data_time_x, data_time_y)
            else:
                cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    colormap = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(frame, 0.7, colormap, 0.3, 0)

    cv2.imshow("Tracking", blend)
    processing_time = time.time() - start_time

    if processing_time > delay:
        frames_to_skip = int(processing_time // delay)

    key = cv2.waitKey(int(delay * 1000))
    if key == ord('q'):
        break
    elif key == ord('s'):
        roi_selected = False
        roi_being_selected = False
        tracker_initialized = False

cap.release()
cv2.destroyAllWindows()
