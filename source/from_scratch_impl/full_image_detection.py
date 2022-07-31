from CascadeClassifier import CascadeClassifier
from utilitaires import read_image
import numpy as np
import cv2

SUBWINDOW_X = 19
SUBWINDOW_Y = 19
FACTOR_STEP = 1

def apply_cascade_to_image(cascade: CascadeClassifier, image_path) -> list:
    image = np.array(read_image(image_path))
    if image[0][0].shape != 0:
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    if image.shape[0] < SUBWINDOW_X or image.shape[1] < SUBWINDOW_Y:
        print("[ERREUR] Impossible de classifier l'image - taille insuffisante :", image.shape)
        return False

    detect_map = np.array([[(False, 0, 0) for j in range(len(image[0]) - SUBWINDOW_X + 1)] for i in range(len(image) - SUBWINDOW_Y + 1)])
    max_factor = min(image.shape[0], image.shape[1])//19

    print("[INFO] Début de l'analyse multi-scalaire")
    for fact in range(1, max_factor + 1, FACTOR_STEP):
        print("     Facteur", fact)
        # réduction de la taille de l'image
        res_image = np.array(cv2.resize(image, (0, 0), fx=1/fact, fy=1/fact, interpolation=cv2.INTER_NEAREST))
        print("     Taille de l'image :", res_image.shape)
        # analyse de chaque région de l'image
        for x in range(len(res_image) - SUBWINDOW_X + 1):
            for y in range(len(res_image[0]) - SUBWINDOW_Y + 1):
                result = cascade.classify(res_image[x:x+SUBWINDOW_X, y:y+SUBWINDOW_Y])
                # indication dans le tableau de détection de la taille de l'objet détecté
                if detect_map[x * fact][y * fact][0] or result:
                    detect_map[x * fact][y * fact] = (True, SUBWINDOW_X*fact, SUBWINDOW_Y*fact)
            
    return detect_map

if __name__ == "__main__":
    cascade = CascadeClassifier.load("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/from_scratch_impl/saves/stop_sign_cascade_1_5_10_50")
    image_path = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_sign_images/stop_signs_images_unprocessed/train/0G8PNL4D4CI0.jpg"
    detect_map = apply_cascade_to_image(cascade, image_path)
    print(detect_map)