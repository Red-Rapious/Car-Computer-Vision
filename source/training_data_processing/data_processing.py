import cv2
import numpy as np
import os

UNPROCESSED_FOLDER = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_signs_images_unprocessed/"
PROCESSED_FOLDER = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_signs_images_processed/"

def format_image(image_name: str, type="train"):
    image = cv2.imread(UNPROCESSED_FOLDER + type + "/" + image_name, cv2.IMREAD_COLOR)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (19, 19), interpolation = cv2.INTER_LINEAR)
        return image
    else:
        print("[ERREUR] L'image '" + image_name +"' n'a pas pu être lue...")
        exit(-1)

def save_image(image, image_name: str, type="train"):
    result = cv2.imwrite(PROCESSED_FOLDER + type + "/" + image_name, image)
    if not result:
        print("[ERREUR] L'image '", image_name, "' n'a pas pu être enregistrée.")

def process_subfolder(type="train"):
    i = 0
    for _, _, files in os.walk(UNPROCESSED_FOLDER + type):
        for file in files:
            if len(file) != 0 and file[0] != ".":
                image = format_image(file, type)
                save_image(image, "stop_sign_" + type + "_" + str(i) + ".pgm", type)
                i += 1

def process_entire_folder():
    print("Démarrage du processing des images")
    print("[INFO] Dossier entrée :", UNPROCESSED_FOLDER)
    print("[INFO] Dossier sortie :", PROCESSED_FOLDER)

    print("Processing en cours des images d'entraînement...")
    process_subfolder("train")
    
    print("Processing en cours des images de test...")
    process_subfolder("test")


if __name__ == "__main__":
    print("[DEBUT DU PROGRAMME]")
    os.mkdir(PROCESSED_FOLDER + "train")
    os.mkdir(PROCESSED_FOLDER + "test")
    process_entire_folder()
    print("[FIN DU PROGRAMME]")