import cv2
import sys
sys.path.insert(0, '/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/yolo')
from object_detector import ObjectDetector
from utilitaires import encadrer_objet
import time

REAL_TIME_MODE = True
SHOW_FPS, SHOW_FRAME, SHOW_RECTANGLES = True, True, True
YOLO, VIOLA = True, True

# Ouverture de la caméra
capture = None
if REAL_TIME_MODE:
   capture = cv2.VideoCapture(0)
else:
   capture = cv2.VideoCapture("ressources/videos/cars_road.mp4")
if not capture.isOpened():
   print("Erreur : la caméra n'est pas allumée'")
   exit(0)

# Importation des classificateurs HAAR
#stop_cascade_classifier = cv2.CascadeClassifier("ressources/classificateurs/Stop_classificateur.xml")
visages_cascade_classifier = cv2.CascadeClassifier("ressources/classificateurs/Visage_classificateur.xml")

# Importation des fichiers YOLO
vd = ObjectDetector("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/yolo/dnn_model/face_yolov3.weights", "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/opencv_impl/yolo/dnn_model/face_yolov3.cfg", None, 1/255)

frame_precendente = time.time()
# Boucle de détection d'image dans la caméra
nb_frames = 0
moyenne_delta = 0
moyenne_fps = 0
while True:
    # IMAGE SOURCE
    rtbool, image = capture.read() # lecture de la caméra
    if not rtbool: # erreur en cas de lecture impossible
        print("Erreur : la VideoCapture n'a pas pu être lue")
        exit(0)

    # VIOLA - JONES
    if VIOLA:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #panneaux_stop = stop_cascade_classifier.detectMultiScale(gray_img, 1.3, 5)
        visages = visages_cascade_classifier.detectMultiScale(gray_img, 1.3, 5)

        # ENCADREMENT DES OBJETS
        if SHOW_RECTANGLES:
            for (x,y,width,height) in visages:
                encadrer_objet(x, y, width, height, image, "Viola-Jones", (0,0,255))

    # YOLO
    if YOLO:
        object_boxes = vd.detect_objects(image)
        
        if SHOW_RECTANGLES:
            for box in object_boxes:
                encadrer_objet(box[0], box[1], box[2], box[3], image, "YOLO", (255, 0, 0))


    # FPS, ETC
    frame_actuelle = time.time()
    delta = frame_actuelle - frame_precendente
    frame_precendente = frame_actuelle

    moyenne_delta = (moyenne_delta * nb_frames + delta) / (nb_frames + 1)
    moyenne_fps = (moyenne_fps * nb_frames + 1/delta) / (nb_frames + 1)

    nb_frames += 1

    #cv2.putText(image, "Delta: " + str(round(delta, 3)) + "s", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    if SHOW_FPS:
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