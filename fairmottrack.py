import cv2
from FairMOT.src.lib.tracker.multitracker import JDETracker

roi_selected = False
roi_being_selected = False
tracker_initialized = False
top_left_pt, bottom_right_pt = [], []

# Initialize FairMOT tracker
tracker = JDETracker(cfg, args.track_thresh)

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

cap = cv2.VideoCapture('match.mp4')
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_roi)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if roi_being_selected:
        cv2.rectangle(frame, top_left_pt[0], (x_cur, y_cur), (0, 255, 0), 2)
        cv2.imshow('Tracking', frame)

    if roi_selected and not tracker_initialized:
        bbox = (top_left_pt[0][0], top_left_pt[0][1], bottom_right_pt[0][0] - top_left_pt[0][0],
                bottom_right_pt[0][1] - top_left_pt[0][1])
        # Initialize FairMOT with the selected region
        tracker.init(frame, bbox)
        tracker_initialized = True

    if tracker_initialized:
        # Update FairMOT tracker
        outputs = tracker.update(frame)
        if len(outputs) > 0:
            for track in outputs:
                bbox = track['bbox']
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(delay)
    if key == ord('q'):
        break
    elif key == ord('s'):
        roi_selected = False
        roi_being_selected = False
        tracker_initialized = False

cap.release()
cv2.destroyAllWindows()
