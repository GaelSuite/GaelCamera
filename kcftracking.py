import cv2
import time

roi_selected = False
roi_being_selected = False
top_left_pt, bottom_right_pt = [], []


def select_roi(event, x_in, y_in, flags, param):
    global top_left_pt, bottom_right_pt, roi_selected, roi_being_selected, x, y
    x, y = x_in, y_in
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



# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("match.mp4")
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_roi)
tracker = cv2.TrackerKCF_create()

tracker_initialized = False

while True:
    time.sleep(0.03)
    ret, frame = cap.read()
    if not ret:
        break

    if roi_being_selected:
        cv2.rectangle(frame, top_left_pt[0], (x, y), (0, 255, 0), 2)
        cv2.imshow('Tracking', frame)

    if roi_selected and not tracker_initialized:
        tracker = cv2.TrackerKCF_create()
        bbox = (top_left_pt[0][0], top_left_pt[0][1], bottom_right_pt[0][0] - top_left_pt[0][0],
                bottom_right_pt[0][1] - top_left_pt[0][1])
        ret = tracker.init(frame, bbox)
        tracker_initialized = True

    if tracker_initialized:
        ret, bbox = tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        roi_selected = False
        roi_being_selected = False
        tracker_initialized = False

cap.release()
cv2.destroyAllWindows()
