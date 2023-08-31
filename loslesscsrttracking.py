import cv2
import numpy as np

# Initialize a frame buffer for averaging
frame_buffer = []

# Function for frame averaging
def frame_average(frame_buffer):
    return np.mean(frame_buffer, axis=0).astype(np.uint8)

# Function for histogram equalization
def apply_histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Function to enhance the image using traditional CV methods and Super-Resolution model
def enhance_image(image):
    img_blur = cv2.GaussianBlur(image, (5, 5), 0)
    img_upscaled = cv2.resize(img_blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img_super_res = apply_super_resolution(img_upscaled)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharp = cv2.filter2D(img_super_res, -1, kernel)
    return img_sharp

# Function to apply Super-Resolution model
def apply_super_resolution(image):
    img_resized = cv2.resize(image, (224, 224))
    img_ycbcr = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
    img_y = img_ycbcr[:,:,0]
    img_y = img_y.astype(np.float32) / 255.0
    img_y = np.expand_dims(np.expand_dims(img_y, axis=0), axis=0)
    net.setInput(img_y)
    enhanced_img_y = net.forward()
    enhanced_img_y = enhanced_img_y.squeeze().clip(0, 1) * 255.0
    enhanced_img_y = enhanced_img_y.astype(np.uint8)
    img_cb = cv2.resize(img_ycbcr[:,:,1], (enhanced_img_y.shape[1], enhanced_img_y.shape[0]))
    img_cr = cv2.resize(img_ycbcr[:,:,2], (enhanced_img_y.shape[1], enhanced_img_y.shape[0]))
    enhanced_img_ycbcr = cv2.merge([enhanced_img_y, img_cb, img_cr])
    enhanced_img_bgr = cv2.cvtColor(enhanced_img_ycbcr, cv2.COLOR_YCrCb2BGR)
    return enhanced_img_bgr

# Initialize variables and load the Super-Resolution model
roi_selected = False
roi_being_selected = False
tracker_initialized = False
top_left_pt, bottom_right_pt = [], []
net = cv2.dnn.readNetFromONNX('super_resolution.onnx')

# Mouse callback function
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

# Initialize video capture and tracker
cap = cv2.VideoCapture('match.mp4')
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', select_roi)
tracker = cv2.TrackerCSRT_create()
fps = cap.get(cv2.CAP_PROP_FPS)
delay = 1
fixed_size = (100, 100)  # You can adjust this size as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if roi_being_selected:
        cv2.rectangle(frame, top_left_pt[0], (x_cur, y_cur), (0, 255, 0), 2)
        cv2.imshow('Tracking', frame)

    if roi_selected and not tracker_initialized:
        tracker = cv2.TrackerCSRT_create()
        bbox = (top_left_pt[0][0], top_left_pt[0][1], bottom_right_pt[0][0] - top_left_pt[0][0], bottom_right_pt[0][1] - top_left_pt[0][1])
        ret = tracker.init(frame, bbox)
        tracker_initialized = True

    if tracker_initialized:
        ret, bbox = tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cropped_frame = frame[p1[1]:p2[1], p1[0]:p2[0]]
            
            # Apply histogram equalization
            cropped_frame = apply_histogram_equalization(cropped_frame)
            
            enhanced_cropped_frame = enhance_image(cropped_frame)
            resized_cropped_frame = cv2.resize(enhanced_cropped_frame, (frame.shape[1], frame.shape[0]))
            cv2.putText(resized_cropped_frame, f"Center: {(p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow('Tracking', resized_cropped_frame)
        else:
            cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow('Tracking', frame)
    else:
        cv2.imshow('Tracking', frame)

    key = cv2.waitKey(delay)
    if key == ord('q'):
        break
    elif key == ord('s'):
        roi_selected = False
        roi_being_selected = False
        tracker_initialized = False

cap.release()
cv2.destroyAllWindows()
