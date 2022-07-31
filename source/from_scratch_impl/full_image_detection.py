from CascadeClassifier import CascadeClassifier
from utilitaires import read_image
import numpy as np

SUBWINDOW_X = 19
SUBWINDOW_Y = 19

def apply_cascade_to_image(cascade: CascadeClassifier, image) -> list:
    image = np.array(image)
    detect_map = np.zeros((len(image) - SUBWINDOW_X + 1, len(image[0]) - SUBWINDOW_Y + 1), dtype=np.uint8)

    for x in range(len(image) - SUBWINDOW_X + 1):
        for y in range(len(image[0]) - SUBWINDOW_Y + 1):
            result = cascade.classify(image[x:x+SUBWINDOW_X][y:y+SUBWINDOW_Y])
            detect_map[x][y] = result
            
    return detect_map

if __name__ == "__main__":
    cascade = CascadeClassifier.load("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/from_scratch_impl/saves/stop_sign_cascade_1_5_10_50")
    image = read_image("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/images/ville_stop.jpg")
    detect_map = apply_cascade_to_image(cascade, image)
    print(detect_map)
