import cv2
import numpy as np

UNPROCESSED_FOLDER = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_signs_images_unprocessed/"
PROCESSED_FOLDER = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_signs_images_processed/"

def format_image(image_name: str, type="train"):
    image = cv2.imread(UNPROCESSED_FOLDER + type + "/" + image_name, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (19, 19), interpolation = cv2.INTER_LINEAR)

    #show_image = cv2.resize(image, (190, 190), interpolation=cv2.INTER_NEAREST)
    #cv2.imshow("Image", show_image)
    #cv2.waitKey(0)
    return image

def save_image(image, image_name: str, type="train"):
    cv2.imwrite(PROCESSED_FOLDER + type + "/" + image_name, image)

if __name__ == "__main__":
    format_image("0G8PNL4D4CI0.jpg", "train")