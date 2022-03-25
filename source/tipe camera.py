# Créé par Mathieu Spiegel

import cv2

cap = cv2.VideoCapture(0)
stop_cascade = cv2.CascadeClassifier('ressources/classificateurs/Stop_classificateur.xml') # classificateur HAAR : IMPORTANT
if not cap.isOpened():
   print("Erreur : la caméra ne fonctionnne pas")
   exit(0)

while True:
   _, img = cap.read()

   
   #img = cv2.resize(img,(340, 220))
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   panneaux = stop_cascade.detectMultiScale(gray, 1.3, 5)
   for (x,y,w,h) in panneaux:
      cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
      # traite le panneau dans une image a part
      #panneau = img[y:y+h, x:x+w]
      #cv2.imshow('panneau STOP', panneau)
   

   cv2.imshow('img', img)

   key = cv2.waitKey(1)
   if key == 27: # touche esc
      break

cap.release()
cv2.destroyAllWindows()
