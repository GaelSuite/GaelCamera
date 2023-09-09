import cv2
import numpy as np
import time

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

selected_points = []
field_dims_in_feet = (12, 12)  # Vex field dimensions in feet

def calculate_acceleration(current_speed, last_speed, time_interval):
    return (current_speed - last_speed) / time_interval

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

def warp_coordinates(point, matrix):
    """Transform a point using the perspective transformation matrix."""
    p = np.array([point[0], point[1], 1])
    warped = np.dot(matrix, p)
    return (warped[0] / warped[2], warped[1] / warped[2])

def calculate_speed(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def predict_trajectory(points, degree=2, length=3):
    if len(points) <= degree:
        return [points[-1]] * length
    x = np.array(range(len(points)))
    y = np.array([p[0] for p in points])
    z = np.array([p[1] for p in points])
    coefficients_x = np.polyfit(x, y, degree)
    coefficients_y = np.polyfit(x, z, degree)
    future_positions_x = [np.polyval(coefficients_x, len(points) + i) for i in range(length)]
    future_positions_y = [np.polyval(coefficients_y, len(points) + i) for i in range(length)]
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

cap = cv2.VideoCapture('match.mp4')
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

# Now, for the main tracking loop
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_roi)
tracker = cv2.TrackerCSRT_create()
fps = cap.get(cv2.CAP_PROP_FPS)
delay = 0.6 / fps
frames_to_skip = 0
last_speed = 0.0
total_distance = 0.0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    if frames_to_skip > 0:
        frames_to_skip -= 1
        continue

    if heatmap is None:
        heatmap = np.zeros_like(frame[:, :, 0]).astype(float)

    if roi_being_selected:
        cv2.rectangle(frame, top_left_pt[0], (x_cur, y_cur), (0, 255, 0), 2)

    if roi_selected and not tracker_initialized:
        tracker = cv2.TrackerCSRT_create()
        bbox = (top_left_pt[0][0], top_left_pt[0][1], bottom_right_pt[0][0] - top_left_pt[0][0],
                bottom_right_pt[0][1] - top_left_pt[0][1])
        ret = tracker.init(frame, bbox)
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
                        cv2.line(frame, tuple(map(int, trajectory[i])), tuple(map(int, trajectory[i+1])), arrow_color, 2)
                    cv2.arrowedLine(frame, tuple(map(int, trajectory[-2])), tuple(map(int, trajectory[-1])), arrow_color, 2, tipLength=0.5)

                box_color = get_tracking_box_color(calculate_speed(past_points[-3], past_points[-2]) if len(past_points) >= 3 else 0, speed_value)
                cv2.rectangle(frame, p1, p2, box_color, 2)
                
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
