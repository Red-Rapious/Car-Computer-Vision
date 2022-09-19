from CascadeClassifier import CascadeClassifier
from utilitaires import read_image
import numpy as np
import cv2
import glob
import os

SUBWINDOW_X = 19
SUBWINDOW_Y = 19
FACTOR_STEP = 1
SHIFT_SCALE = 2
MAX_FACTOR = 10
MIN_FACTOR = 3

FIRST_DETECT_ONLY = True

def apply_cascade_to_image(cascade: CascadeClassifier, image_path) -> list:
    subwindows_nb = 0
    image = np.array(read_image(image_path))
    if image[0][0].shape != 0:
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    if image.shape[0] < SUBWINDOW_X or image.shape[1] < SUBWINDOW_Y:
        print("[ERREUR] Impossible de classifier l'image - taille insuffisante :", image.shape)
        return False

    #detect_map = np.array([[(False, 0, 0) for j in range(len(image[0]) - SUBWINDOW_X + 1)] for i in range(len(image) - SUBWINDOW_Y + 1)])
    boxes =  []

    print("[INFO] Début de l'analyse multi-scalaire de l'image :", image_path, "de taille", image.shape)
    max_factor = min(min(image.shape[0], image.shape[1])//19, MAX_FACTOR)
    detected = False
    for fact in reversed(range(MIN_FACTOR, max_factor + 1, FACTOR_STEP)):
        
        # réduction de la taille de l'image
        res_image = np.array(cv2.resize(image, (0, 0), fx=1/fact, fy=1/fact, interpolation=cv2.INTER_NEAREST))
        print("     Facteur :", fact, "     Taille de l'image :", res_image.shape)
        
        # analyse de chaque région de l'image
        for x in range(0, len(res_image) - SUBWINDOW_X + 1, SHIFT_SCALE):
            for y in range(0, len(res_image[0]) - SUBWINDOW_Y + 1, SHIFT_SCALE):
                xmin, xmax, ymin, ymax = x, x+SUBWINDOW_X, y, y+SUBWINDOW_Y
                subwindows_nb += 1
                result = cascade.classify(res_image[xmin:xmax, ymin:ymax])
                
                # indication dans le tableau de détection de la taille de l'objet détecté
                if result:
                    boxes.append([ymin*fact, xmin*fact, SUBWINDOW_Y*fact, SUBWINDOW_X*fact, (0, 255, 0)])
                    if FIRST_DETECT_ONLY:
                        print("     Analysed subwindows:", subwindows_nb)
                        return boxes
            
    print("     Analysed subwindows:", subwindows_nb)
    return boxes

def encadrer_objet(x: int, y: int, width: int, height: int, image, texte: str = "", couleur=(0,255,0)):
    """ Fonction encadrant d'un carré vert un objet dans une image, qui a été précédement détecté """
    cv2.rectangle(image, (x,y), (x+width,y+height), couleur, 2)
    if texte != "":
        global_size = (width+height)/2 # facteur global indiquant la taille de l'image
        cv2.putText(image, texte, (x+int(global_size/5.5),y-int(global_size/20)), cv2.FONT_HERSHEY_DUPLEX, global_size/340, couleur, 2, cv2.LINE_AA)

if __name__ == "__main__":
    cascade = CascadeClassifier.load("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/from_scratch_impl/saves/stop_sign_v2_cascade_1_5")
    #image_path = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_sign_images/stop_signs_images_unprocessed/train/0G8PNL4D4CI0.jpg"
    #image_path = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/full_test_images/5MXRDZVI05NT.jpg"
    images_folder = glob.glob("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/full_test_images/*.jpg")
    
    for image_path in images_folder:
        cv2.destroyAllWindows()
        boxes = apply_cascade_to_image(cascade, image_path)

        image = cv2.imread(image_path)
        for box in boxes:
            encadrer_objet(box[0], box[1], box[2], box[3], image, "", box[4])
        cv2.imshow(os.path.basename(image_path), image)
        cv2.waitKey(0)