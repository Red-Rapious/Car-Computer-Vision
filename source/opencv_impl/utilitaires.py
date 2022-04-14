import cv2

def encadrer_objet(x: int, y: int, width: int, height: int, image, texte: str, couleur=(0,255,0)):
   """ Fonction encadrant d'un carré vert un objet dans une image, qui a été précédement détecté """
   cv2.rectangle(image, (x,y), (x+width,y+height), couleur, 2)
   global_size = (width+height)/2 # facteur global indiquant la taille de l'image
   cv2.putText(image, texte, (x+int(global_size/5.5),y-int(global_size/20)), cv2.FONT_HERSHEY_DUPLEX, global_size/340, couleur, 2, cv2.LINE_AA)