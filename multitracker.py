import cv2

roi_selected = False
roi_being_selected = False
tracker_initialized = False
top_left_pt, bottom_right_pt = [], []

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

# Capture video stream (use 0 for default camera)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("match.mp4")

cv2.namedWindow('Live Stream')
cv2.setMouseCallback('Live Stream', select_roi)
multi_tracker = cv2.legacy.MultiTracker.create()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if roi_being_selected:
        cv2.rectangle(frame, top_left_pt[0], (x_cur, y_cur), (0, 255, 0), 2)

    if roi_selected and not tracker_initialized:
        bbox = (top_left_pt[0][0], top_left_pt[0][1], bottom_right_pt[0][0] - top_left_pt[0][0],
                bottom_right_pt[0][1] - top_left_pt[0][1])
        multi_tracker.add(cv2.legacy.TrackerCSRT.create(), frame, bbox) #8/10, loses when corners
        #multi_tracker.add(cv2.legacy.TrackerMOSSE.create(), frame, bbox) #3/10, loses when goal
        #multi_tracker.add(cv2.legacy.TrackerKCF.create(), frame, bbox) #-2/10 straight ass
        #multi_tracker.add(cv2.legacy.TrackerMIL.create(), frame, bbox) #4/10, issues with speed and switching bot
        #multi_tracker.add(cv2.legacy.TrackerBoosting.create(), frame, bbox) #4/10 accurate but switches bot
        #multi_tracker.add(cv2.legacy.TrackerMedianFlow.create(), frame, bbox) #2/10 fast but ass
        #multi_tracker.add(cv2.legacy.TrackerTLD.create(), frame, bbox) #4/10 good at multi but bad tracking
        tracker_initialized = True

    if tracker_initialized:
        ret, boxes = multi_tracker.update(frame)
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

    cv2.imshow('Live Stream', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        roi_selected = False
        roi_being_selected = False
        tracker_initialized = False

cap.release()
cv2.destroyAllWindows()
