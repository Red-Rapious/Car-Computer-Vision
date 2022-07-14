import cv2
import numpy as np
from vehicle_detector import VehicleDetector
import time
import sys

sys.path.insert(0, '/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/')
from utilitaires import encadrer_objet

REAL_TIME_MODE = True

# Ouverture de la caméra
capture = None
if REAL_TIME_MODE:
   capture = cv2.VideoCapture(0)
else:
   capture = cv2.VideoCapture("ressources/videos/Parc_naturel.mp4")
if not capture.isOpened():
   print("Erreur : la caméra n'est pas allumée'")
   exit(0)

vd = VehicleDetector()
frame_precendente = time.time()

while True:
    rtbool, image = capture.read() # lecture de la caméra
    if not rtbool: # erreur en cas de lecture impossible
      print("Erreur : la VideoCapture n'a pas pu être lue")
      exit(0)

    # Application du réseau de neurones pour détecter les véhicules
    vehicle_boxes = vd.detect_vehicles(image)
    for box in vehicle_boxes:
        encadrer_objet(box[0], box[1], box[2], box[3], image, "Vehicule")

    frame_actuelle = time.time()
    delta = frame_actuelle - frame_precendente
    frame_precendente = frame_actuelle

    #cv2.putText(image, "Delta: " + str(round(delta, 3)) + "s", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "FPS: " + str(round(1/delta, 3)), (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Camera", image) # affichage de l'image légèrement modifiée par l'ajout de texte et rectangles

    # Sortie de boucle si la touche ESC est pressée
    key = cv2.waitKey(1)
    if key == 27:
        break

# Fermeture de la caméra et des fenêtres
capture.release()
cv2.destroyAllWindows()
