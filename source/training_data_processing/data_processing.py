import cv2
import numpy as np
import os

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

def process_entire_folder():
    print("Démarrage du processing des images")
    print("[INFO] Dossier entrée :", UNPROCESSED_FOLDER)
    print("[INFO] Dossier sortie :", PROCESSED_FOLDER)

    print("Processing en cours des images d'entraînement...")
    i = 0
    for _, _, files in os.walk(UNPROCESSED_FOLDER + "train"):
        for file in files:
            image = format_image(file, "train")
            save_image(image, "stop_sign_train_" + str(i), "train")
            i += 1
    
    print("Processing en cours des images de test...")
    i = 0
    for _, _, files in os.walk(UNPROCESSED_FOLDER + "test"):
        for file in files:
            image = format_image(file, "test")
            save_image(image, "stop_sign_test_" + str(i), "test")
            i += 1


if __name__ == "__main__":
    print("[DEBUT DU PROGRAMME]")
    process_entire_folder()
    print("[FIN DU PROGRAMME]")