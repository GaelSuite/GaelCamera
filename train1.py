from roboflow import Roboflow

rf = Roboflow(api_key="OxsAN6ixRjwybe4l6akJ")
project = rf.workspace("gaelcam").project("gaelcam")
version = project.version(1)

dataset = version.download("yolov8")

from ultralytics import YOLO

import subprocess

dataset_location = dataset.location


command = f"yolo train model=yolov8m.pt data={dataset_location}/data.yaml epochs=50 imgsz=640"
 

subprocess.run(command, shell=True, check=True)
