import cv2
import numpy as np

def format_image(image_path: str):
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (19, 19), interpolation = cv2.INTER_LINEAR)

    show_image = cv2.resize(image, (190, 190), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Image", show_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    format_image("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_signs_images_unprocessed/data/test/stop_sign/gray-01MB90O9N0J2.jpg")