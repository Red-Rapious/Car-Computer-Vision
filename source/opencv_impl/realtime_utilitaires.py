import cv2

def encadrer_objet(x: int, y: int, width: int, height: int, image, texte: str, couleur=(0,255,0)):
   """ Fonction encadrant d'un carré vert un objet dans une image, qui a été précédement détecté """
   cv2.rectangle(image, (x,y), (x+width,y+height), couleur, 2)
   global_size = (width+height)/2 # facteur global indiquant la taille de l'image
   cv2.putText(image, texte, (x+int(global_size/5.5),y-int(global_size/20)), cv2.FONT_HERSHEY_DUPLEX, global_size/340, couleur, 2, cv2.LINE_AA)

   distance = calculer_distance(width)
   cv2.putText(image, f'distance : {round(distance, 2)} m', (x+int(global_size/6),y+height+int(global_size/10) - 20), cv2.FONT_HERSHEY_DUPLEX, global_size/340, (0, 0, 255))

def calculer_distance(width: int):
   """ Fonction calculant la distance entre la caméra et l'objet à détecter """
   screen_width = 190   # en pixel, taille sur l'écran de l'objet pendant la référence
   ref_dist = 30     # en cm, distance entre la caméra et l'objet pendant la référence
   real_width = 5.8  # en cm, vraie taille de l'objet pendant la référence

   detected_object_width = 8
   
   focal = (screen_width * ref_dist) / real_width
   distance = ((detected_object_width * focal)/width)*(10**-2)  # distance en mètres du panneau
   return distance