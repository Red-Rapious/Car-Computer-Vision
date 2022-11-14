import cv2
import numpy as np
from object_detector import ObjectDetector
import time
import sys
sys.path.insert(0, '/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/')
from realtime_utilitaires import encadrer_objet

REAL_TIME_MODE = True
SHOW_FPS, SHOW_FRAME, SHOW_RECTANGLES = True, True, True

# Ouverture de la caméra
capture = None
if REAL_TIME_MODE:
   capture = cv2.VideoCapture(0)
else:
   capture = cv2.VideoCapture("ressources/videos/Parc_naturel.mp4")
if not capture.isOpened():
   print("Erreur : la caméra n'est pas allumée'")
   exit(0)

vd = ObjectDetector("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/yolo/dnn_model/face_yolov3.weights", "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/yolo/dnn_model/face_yolov3.cfg", None, 1/255)
frame_precendente = time.time()

nb_frames = 0
moyenne_delta = 0
moyenne_fps = 0
while True:
   rtbool, image = capture.read() # lecture de la caméra
   if not rtbool: # erreur en cas de lecture impossible
      print("Erreur : la VideoCapture n'a pas pu être lue")
      exit(0)

   # Application du réseau de neurones pour détecter les véhicules
   object_boxes = vd.detect_objects(image)
   if SHOW_RECTANGLES:
      for box in object_boxes:
         encadrer_objet(box[0], box[1], box[2], box[3], image, "Visage")

   frame_actuelle = time.time()
   delta = frame_actuelle - frame_precendente
   frame_precendente = frame_actuelle

   moyenne_delta = (moyenne_delta * nb_frames + delta) / (nb_frames + 1)
   moyenne_fps = (moyenne_fps * nb_frames + 1/delta) / (nb_frames + 1)

   nb_frames += 1

   if SHOW_FPS:
      #cv2.putText(image, "Delta: " + str(round(delta, 3)) + "s", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(image, "FPS: " + str(round(1/delta, 3)), (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

   if SHOW_FRAME:
      cv2.imshow("Camera", image) # affichage de l'image légèrement modifiée par l'ajout de texte et rectangles

   # Sortie de boucle si la touche ESC est pressée
   key = cv2.waitKey(1)
   if key == 27:
      break

   if not SHOW_FRAME and nb_frames > 100:
      break

# Fermeture de la caméra et des fenêtres
capture.release()
cv2.destroyAllWindows()
print("[RESULTATS]")
print("     Delta moyen: " + str(round(moyenne_delta, 3)) + "s")
print("     FPS moyen: " + str(round(moyenne_fps, 3)))