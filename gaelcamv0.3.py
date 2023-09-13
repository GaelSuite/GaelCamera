import cv2
import numpy as np
import time
import statistics

class TrackerConfig:
    MAX_SPEED = 150  # or whatever value you want
    DEGREE = 2
    MAX_PAST_POINTS = 10
    PREDICTION_LENGTH = 3
    SPEED_THRESHOLD = 5
    FIELD_DIMS_IN_FEET = (12, 12)  # Vex field dimensions in feet

class TrackerUtils:
    @staticmethod
    def remove_outliers(data):
        sorted_data = sorted(data)
        
        Q1 = statistics.median(sorted_data[:len(sorted_data) // 2])
        Q3 = statistics.median(sorted_data[len(sorted_data) // 2:])

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return [x for x in data if lower_bound <= x <= upper_bound]
        
    @staticmethod
    def calculate_acceleration(current_speed, last_speed, time_interval):
        return (current_speed - last_speed) / time_interval
        
    @staticmethod
    def warp_coordinates(point, matrix):
        """Transform a point using the perspective transformation matrix."""
        p = np.array([point[0], point[1], 1])
        warped = np.dot(matrix, p)
        return (warped[0] / warped[2], warped[1] / warped[2])
    
    @staticmethod
    def calculate_speed(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
        
    @staticmethod
    def predict_trajectory(points, degree=TrackerConfig.DEGREE, length=TrackerConfig.PREDICTION_LENGTH):
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
    
    @staticmethod
    def get_speed_color(start_point, end_point):
        dist = np.linalg.norm(np.array(start_point) - np.array(end_point))
        max_speed = 100
        ratio = np.clip(dist / max_speed, 0, 1)
        r = int(255 * ratio)
        g = int(255 * (1 - abs(2 * ratio - 1)))
        b = int(255 * (1 - ratio))
        return (b, g, r)
        
    @staticmethod
    def get_tracking_box_color(last_speed, current_speed):
        speed_change = abs(current_speed - last_speed)
        if speed_change < 5:
            return (0, 255, 0)
        elif 5 <= speed_change < 15:
            return (0, 255, 255)
        else:
            return (0, 0, 255)

class RobotTracker:
    def __init__(self):
        self.roi_selected = False
        self.roi_being_selected = False
        self.tracker_initialized = False
        self.top_left_pt, self.bottom_right_pt = [], []
        self.heatmap = None
        self.past_points = []
        self.selected_points = []
        self.temp_frame = None
        self.frames_to_skip = 0
        self.delay = 1.0 / 60
        self.x_cur = None
        self.y_cur = None
        self.frame = None
        self.max_past_points = 10
        self.cap = None
        

    def select_roi(self, event, x, y, flags, param):
        self.x_cur, self.y_cur = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.roi_being_selected:
                self.roi_selected = False
                self.top_left_pt = [(x, y)]
                self.roi_being_selected = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.bottom_right_pt = [(x, y)]
            self.roi_selected = True
            self.roi_being_selected = False
            cv2.rectangle(self.frame, self.top_left_pt[0], self.bottom_right_pt[0], (0, 255, 0), 2)
            cv2.imshow('Tracking', self.frame)

        
    def get_field_corners(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            cv2.circle(self.temp_frame, (x, y), 5, (0, 0, 255), -1)  # Draw red dot
            if len(self.selected_points) > 1:
                cv2.line(self.temp_frame, self.selected_points[-2], self.selected_points[-1], (255, 0, 0), 2)  # Draw blue line
            cv2.imshow('Select Field Corners', self.temp_frame)
            if len(self.selected_points) == 4:
                cv2.line(self.temp_frame, self.selected_points[-1], self.selected_points[0], (255, 0, 0), 2)  # Close the polygon
                cv2.imshow('Select Field Corners', self.temp_frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

    
    def display_ar_overlays(self, frame, center, speed, acceleration, total_distance, robot_id="R001"):
        text_color = (0, 255, 255)  # Yellow
        cv2.putText(self.frame, f"Speed: {speed:.2f} ft/s", (center[0] + 20, center[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(self.frame, f"Acceleration: {acceleration:.2f} ft/s^2", (center[0] + 20, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(self.frame, f"Total Distance: {total_distance:.2f} ft", (center[0] + 20, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(self.frame, f"ID: {robot_id}", (center[0] + 20, center[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


    def track(self):
        cap = cv2.VideoCapture('match.mp4')
        
        # Setup windows and mouse callbacks
        cv2.namedWindow('Select Field Corners')
        cv2.setMouseCallback('Select Field Corners', self.get_field_corners)
        
        ret, self.temp_frame = cap.read()
        while len(self.selected_points) < 4:
            cv2.imshow('Select Field Corners', self.temp_frame)
            cv2.waitKey(1)
        
        # Perspective transformation matrix
        dst_points = np.array([
            [0, 0],
            [TrackerConfig.FIELD_DIMS_IN_FEET[0], 0],
            [TrackerConfig.FIELD_DIMS_IN_FEET[0], TrackerConfig.FIELD_DIMS_IN_FEET[1]],
            [0, TrackerConfig.FIELD_DIMS_IN_FEET[1]]
        ], dtype="float32")
        matrix = cv2.getPerspectiveTransform(np.array(self.selected_points, dtype="float32"), dst_points)
        
        self.cap = cv2.VideoCapture('match.mp4')
        cv2.namedWindow('Tracking')
        cv2.setMouseCallback('Tracking', self.select_roi)
        last_speed = 0.0
        total_distance = 0.0
        max_score = 0
        
        total_offensive_score = []
        total_defensive_score = []
        
        while True:
            start_time = time.time()
            ret, self.frame = self.cap.read()
            if not ret:
                break

            if self.frames_to_skip > 0:
                self.frames_to_skip -= 1
                continue

            if self.heatmap is None:
                self.heatmap = np.zeros_like(self.frame[:, :, 0]).astype(float)

            if self.roi_being_selected and self.top_left_pt:
                cv2.rectangle(self.frame, self.top_left_pt[0], (self.x_cur, self.y_cur), (0, 255, 0), 2)


            if self.roi_selected and not self.tracker_initialized:
                tracker = cv2.TrackerCSRT_create()
                bbox = (self.top_left_pt[0][0], self.top_left_pt[0][1], self.bottom_right_pt[0][0] - self.top_left_pt[0][0],
                        self.bottom_right_pt[0][1] - self.top_left_pt[0][1])
                ret = tracker.init(self.frame, bbox)
                self.tracker_initialized = True


            if self.tracker_initialized:
                ret, bbox = tracker.update(self.frame)
                if ret:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
                    
                    self.heatmap[p1[1]:p2[1], p1[0]:p2[0]] += 1
                    
                    self.past_points.append(center)
                    if len(self.past_points) > self.max_past_points:
                        self.past_points.pop(0)

                    center_real_world = TrackerUtils.warp_coordinates(center, matrix)
                    if len(self.past_points) >= 2:
                        speed_value = TrackerUtils.calculate_speed(TrackerUtils.warp_coordinates(self.past_points[-2], matrix), center_real_world) / self.delay
                        acceleration = TrackerUtils.calculate_acceleration(speed_value, last_speed, self.delay)
                        total_distance += speed_value * self.delay
                        last_speed = speed_value
                        self.display_ar_overlays(self.frame, center, speed_value, acceleration, total_distance)
                        
                        if speed_value > TrackerConfig.SPEED_THRESHOLD:
                            trajectory = TrackerUtils.predict_trajectory(self.past_points, TrackerConfig.DEGREE, TrackerConfig.PREDICTION_LENGTH)

                            arrow_color = TrackerUtils.get_speed_color(center, trajectory[-1])
                            for i in range(len(trajectory) - 1):
                                cv2.line(self.frame, tuple(map(int, trajectory[i])), tuple(map(int, trajectory[i+1])), arrow_color, 2)
                            cv2.arrowedLine(self.frame, tuple(map(int, trajectory[-2])), tuple(map(int, trajectory[-1])), arrow_color, 2, tipLength=0.5)

                        box_color = TrackerUtils.get_tracking_box_color(TrackerUtils.calculate_speed(self.past_points[-3], self.past_points[-2]) if len(self.past_points) >= 3 else 0, speed_value)

                        cv2.rectangle(self.frame, p1, p2, box_color, 2)
                        
                        speed_score = speed_value / TrackerConfig.MAX_SPEED
                        acceleration_score = max(0, acceleration)  # consider only positive acceleration for offense
                        distance_score = total_distance / (TrackerConfig.FIELD_DIMS_IN_FEET[0] * TrackerConfig.FIELD_DIMS_IN_FEET[1])

                        offensive_half_activity = np.mean(self.heatmap[:, self.heatmap.shape[1]//2:]) / 255  # normalized between 0 and 1
                        
                        # Defensive Scoring
                        deceleration_score = max(0, -acceleration)  # consider only negative acceleration for defense
                        defensive_half_activity = np.mean(self.heatmap[:, :self.heatmap.shape[1]//2]) / 255  # normalized between 0 and 1
                        direction_change_score = np.std(self.past_points) / self.max_past_points  # Standard deviation of past points as a measure of direction change
                        
                        # Weighted Scoring (Weights are hypothetical and might need tuning)
                        offensive_score = ((0.3 * speed_score + 0.2 * acceleration_score + 0.3 * distance_score + 0.2 * offensive_half_activity) * 100)
                        defensive_score = ((0.3 * deceleration_score + 0.4 * defensive_half_activity + 0.3 * direction_change_score) * 100)
                        if offensive_score > max_score:
                            max_score = offensive_score
                        elif defensive_score > max_score:
                            max_score = defensive_score
                            
                        total_offensive_score.append(offensive_score)
                        total_defensive_score.append(defensive_score)
                        
                        # Display Scores
                        cv2.putText(self.frame, f"Offensive Score: {offensive_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(self.frame, f"Defensive Score: {defensive_score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    else:
                        cv2.putText(self.frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            self.heatmap = cv2.GaussianBlur(self.heatmap, (15, 15), 0)
            heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
            colormap = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            blend = cv2.addWeighted(self.frame, 0.7, colormap, 0.3, 0)

            cv2.imshow("Tracking", blend)
            processing_time = time.time() - start_time

            if processing_time > self.delay:
                frames_to_skip = int(processing_time // self.delay)

            key = cv2.waitKey(int(self.delay * 1000))

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.roi_selected = False
                self.roi_being_selected = False
                self.tracker_initialized = False
        offense_score = statistics.mean(TrackerUtils.remove_outliers(total_offensive_score))
        defense_score = statistics.mean(TrackerUtils.remove_outliers(total_defensive_score))
        
        print(f"Offense: {(defense_score/max_score)*100*10}")
        print(f"Defense: {(offense_score/max_score)*100*10}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = RobotTracker()
    tracker.track()

