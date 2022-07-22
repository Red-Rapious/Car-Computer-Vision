import cv2

# Ouverture de la caméra
capture = cv2.VideoCapture(0)

# Importation des classificateurs HAAR
stop_cascade_classifier = cv2.CascadeClassifier("ressources/classificateurs/Stop_classificateur.xml")

# Boucle de détection d'image dans la caméra
while True:
   _, image = capture.read() # lecture de la caméra

   # Application des filtres HAAR pour rechercher les panneaux
   gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   panneaux_stop = stop_cascade_classifier.detectMultiScale(gray_img, 1.3, 5)

   # Pour chaque panneau stop, on dessine un rectangle sur l'image et on ajoute du texte
   for (x,y,width,height) in panneaux_stop:
      encadrer_objet(x, y, width, height, image, "Panneau stop", (255, 0, 0))

   cv2.imshow("Camera", image) # affichage de l'image légèrement modifiée par l'ajout de texte et rectangles