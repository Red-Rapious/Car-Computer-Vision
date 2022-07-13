import cv2
from utilitaires import encadrer_objet
import time

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

# Importation des classificateurs HAAR
stop_cascade_classifier = cv2.CascadeClassifier("ressources/classificateurs/Stop_classificateur.xml")
visages_cascade_classifier = cv2.CascadeClassifier("ressources/classificateurs/Visage_classificateur.xml")
# Le fichier HAAR récupéré sur internet pour la détection de piétons semble être de mauvaise qualité
#pietons_cascade_classifier = cv2.CascadeClassifier("ressources/classificateurs/Pieton_classificateur.xml")
voitures_cascade_classifier = cv2.CascadeClassifier("ressources/classificateurs/Voitures_classificateur.xml")
feux_cascade_classifier = cv2.CascadeClassifier("ressources/classificateurs/Feu_classificateur.xml")

frame_precendente = time.time()
# Boucle de détection d'image dans la caméra
while True:
   rtbool, image = capture.read() # lecture de la caméra
   if not rtbool: # erreur en cas de lecture impossible
      print("Erreur : la VideoCapture n'a pas pu être lue")
      exit(0)

   # Application des filtres HAAR pour rechercher les panneaux
   gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   panneaux_stop = stop_cascade_classifier.detectMultiScale(gray_img, 1.3, 5)
   visages = visages_cascade_classifier.detectMultiScale(gray_img, 1.3, 5)
   #pietons = pietons_cascade_classifier.detectMultiScale(gray_img, 1.3, 45)
   voitures = voitures_cascade_classifier.detectMultiScale(gray_img, 1.3, 4)
   feux = feux_cascade_classifier.detectMultiScale(gray_img, 1.3, 4)

   # Pour chaque panneau stop, on dessine un rectangle sur l'image et on ajoute du texte
   for (x,y,width,height) in panneaux_stop:
      encadrer_objet(x, y, width, height, image, "Panneau stop", (255, 0, 0))
   for (x,y,width,height) in visages:
      encadrer_objet(x, y, width, height, image, "Visage")
   #for (x,y,width,height) in pietons:
   #   encadrer_objet(x, y, width, height, image, "Pieton")
   for (x,y,width,height) in voitures:
      encadrer_objet(x, y, width, height, image, "Voiture", (255, 0, 0))
   for (x,y,width,height) in feux:
      encadrer_objet(x, y, width, height, image, "Feu")

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