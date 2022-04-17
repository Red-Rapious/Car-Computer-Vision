from operator import xor
import matplotlib.pyplot as plt
import numpy as np
from ViolaJones import ViolaJones
from WeakClassifier import WeakClassifier
from RectangleRegion import RectangleRegion
import cv2
from time import sleep

def features_count_graph(start: int, stop: int, step: int, ratio: float=16/9) -> None:
    """ Trace le nombre de features possibles contenues
    pour des images verticales de hauteurs allant de 'start' à 'stop' 
    avec un pas de 'step' et ayant un ratio 'ratio' """
    vj = ViolaJones(0)
    
    side = [i for i in range(start, stop, step)]
    nb_features = [len(vj.build_features((s, int(s/ratio)))) for s in side]

    plt.scatter(side, nb_features)
    plt.xlabel("Côté de l'image (px)")
    plt.ylabel("Nombre de features")
    plt.title("Nombre de features possibles en fonction de la taille de l'image")
    plt.show()

#features_count_graph(10, 30, 2)

def draw_rectangle(image, angle1: tuple, angle2: tuple, is_positive: bool) -> None:
    col = (255, 255, 255) if is_positive else (0, 0, 0)
    for x in range(angle1[0], angle2[0] + 1):
        for y in range(angle1[1], angle2[1] + 1):
            image[y][x] = col

    #cv2.rectangle(image, angle1, angle2, col, 2)


def draw_weakclassifier_on_image(image: list, weak_classifier: WeakClassifier) -> None:
    """ Dessine la feature d'un WeakClassifier sur une image avec OpenCV """
    is_polarity_pos = True if weak_classifier.polarity == 1 else False
    for region in weak_classifier.positive_regions:
        draw_rectangle(image, (region.x, region.y), (region.x + region.width, region.y + region.height), is_polarity_pos)
    for region in weak_classifier.negative_regions:
        draw_rectangle(image, (region.x, region.y), (region.x + region.width, region.y + region.height), not is_polarity_pos)


def main() -> None:
    """ Programme de test """
    capture = cv2.VideoCapture("ressources/videos/Parc_naturel.mp4")
    classifier = WeakClassifier([RectangleRegion(100, 100, 50, 50), RectangleRegion(150, 150, 50, 50)], [RectangleRegion(150, 100, 50, 50), RectangleRegion(100, 150, 50, 50)], 0.0, -1)
    while True:
        rtbool, image = capture.read()
        draw_weakclassifier_on_image(image, classifier)

        cv2.imshow("Image", image)

        # Sortie de boucle si la touche ESC est pressée
        key = cv2.waitKey(1)
        if key == 27:
            break