import cv2

# Ouverture de la caméra
capture = cv2.VideoCapture(0)
if not capture.isOpened():
   print("Erreur : la caméra n'est pas allumée'")
   exit(0)

# Importation du classificateur HAAR
cascade_classifier = cv2.CascadeClassifier("ressources/classificateurs/Stop_classificateur.xml")

# Boucle de détection d'image dans la caméra
while True:
   rtbool, image = capture.read() # lecture de la caméra
   if not rtbool: # erreur en cas de lecture impossible
      print("Erreur : la VideoCapture n'a pas pu être lue")
      exit(0)

   # Application des filtres HAAR
   gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   panneaux_stop = cascade_classifier.detectMultiScale(gray_img, 1.3, 5)

   for (x,y,width,height) in panneaux_stop:
      cv2.rectangle(image, (x,y), (x+width,y+height), (0,255,0), 2)
      global_size = (width+height)/2
      img_text = cv2.putText(image, "Panneau stop", (x+int(global_size/5.5),y-int(global_size/20)), cv2.FONT_HERSHEY_SIMPLEX, global_size/340, (0,255,0), 2, cv2.LINE_AA)


   cv2.imshow("Camera", image)

   # Sortie de boucle si la touche ESC est pressée
   key = cv2.waitKey(1)
   if key == 27:
      break

# Fermeture de la caméra et des fenêtres
capture.release()
cv2.destroyAllWindows()