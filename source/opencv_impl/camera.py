# Créé par Mathieu Spiegel

import cv2

class Camera:
   def __init__(self):

      self.cap = cv2.VideoCapture(0)  # On choisi la 1 caméra : peut etre 0
      self.stop_cascade = cv2.CascadeClassifier('ressources/classificateurs/Stop_classificateur.xml')  # classificateur HAAR : IMPORTANT
      #self.car_cascade = cv2.CascadeClassifier('xml/cars.xml')
      
      # style ecriture
      self.font = cv2.FONT_HERSHEY_COMPLEX


   def get_image(self):
      """ capture du flux video """
      _, self.frame = self.cap.read() # recupere l'image

   def display(self):
      """ affichage du retour camera """
      cv2.imshow('Camera', self.frame)

   def stop_detection(self):
      """ detection de panneaux stop """
      gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
      panneaux = self.stop_cascade.detectMultiScale(gray, 1.3, 5)
      distance = None
      for (x, y, w, h) in panneaux:
         cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
         
         w_ref_img = 190   # en pixel taille sur l'ecran de l'objet pendant la reference
         dist_ref = 30     # en cm distance entre la cam et l'objet pendant la reference
         real_width = 5.8  # en cm vrai taille de l'objet pendant la reference
          
         focal = (w_ref_img* dist_ref)/ real_width
         distance = ((real_width * focal)/w  )*10**-2  # distance en metre du panneau

         cv2.putText(self.frame, f'distance : {round(distance, 2)} m', (20, 20), self.font, 0.6, (0, 0, 255))
         

      return ("STOP", distance)

   def car_detection(self):
      """ detection de voitures """
      gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
      voitures = self.car_cascade.detectMultiScale(gray, 1.1, 2)
      for (x, y, w, h) in voitures:
         cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      


if __name__ == "__main__":
   Cam = Camera()
   while True:
      Cam.get_image()
      Cam.stop_detection()
      Cam.display()

      key = cv2.waitKey(1)
      if key == 27: # touche esc
         break
      
   Cam.cap.release()
   cv2.destroyAllWindows()
