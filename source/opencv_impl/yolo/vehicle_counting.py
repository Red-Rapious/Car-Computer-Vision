import cv2
import glob
from object_detector import ObjectDetector
import time

scale_percent = 50

# Load Veichle Detector
vd = ObjectDetector("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/yolo/dnn_model/vehicle_yolov4.cfg", "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/yolo/dnn_model/vehicle_yolov4.weights", None, 1/255)

# Load images from a folder
images_folder = glob.glob("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/yolo/images/*.jpg")

vehicles_folder_count = 0

# Loop through all the images
for img_path in images_folder:
    #print("Img path", img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)))

    t_start = time.time()
    vehicle_boxes = vd.detect_objects(img)
    vehicle_count = len(vehicle_boxes)
    print("Temps :", time.time() - t_start, "sec.")

    # Update total count
    vehicles_folder_count += vehicle_count

    for box in vehicle_boxes:
        x, y, w, h = box

        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)

        cv2.putText(img, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

    cv2.imshow("Cars", img)
    cv2.waitKey(1)

print("Total current count", vehicles_folder_count)