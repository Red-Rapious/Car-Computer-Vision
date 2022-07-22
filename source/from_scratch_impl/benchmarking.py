import matplotlib.pyplot as plt
import numpy as np
from ViolaJones import ViolaJones
from WeakClassifier import WeakClassifier
from RectangleRegion import RectangleRegion
from utilitaires import integral_image, read_image
import cv2
import pandas

def features_count_graph(start: int, stop: int, step: int, ratio: float=16/9, show_special_point: bool=False, special_point: int=19) -> None:
    """ Trace le nombre de features possibles contenues
    pour des images verticales de hauteurs allant de 'start' à 'stop' 
    avec un pas de 'step' et ayant un ratio 'ratio' """
    vj = ViolaJones(0)
    
    side = [i for i in range(start, stop, step)]
    nb_features = [len(vj.build_features((s, int(s/ratio)))) for s in side]

    plt.figure(" ")
    plt.scatter(side, nb_features)
    if show_special_point:
        plt.scatter([special_point], [len(vj.build_features((special_point, int(special_point/ratio))))], c="red")
    plt.xlabel("Côté de l'image (px)")
    plt.ylabel("Nombre de features")
    plt.title("Nombre de features possibles en fonction de la taille de l'image")
    plt.show()

def fill_rectangle(image, angle1: tuple, angle2: tuple, is_positive: bool) -> None:
    col = (255, 255, 255) if is_positive else (0, 0, 0)
    for x in range(angle1[0], angle2[0] + 1):
        for y in range(angle1[1], angle2[1] + 1):
            image[y][x] = col


def draw_weakclassifier_on_image(image: list, weak_classifier: WeakClassifier) -> None:
    """ Dessine la feature d'un WeakClassifier sur une image avec OpenCV """
    is_polarity_pos = True if weak_classifier.polarity == 1 else False
    for region in weak_classifier.positive_regions:
        fill_rectangle(image, (region.x, region.y), (region.x + region.width, region.y + region.height), is_polarity_pos)
    for region in weak_classifier.negative_regions:
        fill_rectangle(image, (region.x, region.y), (region.x + region.width, region.y + region.height), not is_polarity_pos)

def show_image_as_table(image: list, title: str) -> None:
    #define figure and axes
    fig, ax = plt.subplots(num=title)

    #hide the axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    #create data
    df = pandas.DataFrame(image)

    #create table
    table = ax.table(cellText=df.values, loc='center')

    #display table
    fig.tight_layout()
    plt.show()

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

if __name__ == "__main__":
    image = np.array(read_image("ressources/faces_images/train/face/face00324.pgm"), dtype=np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Image en nuances de gris", cv2.resize(image, (0, 0), fx=30, fy=30, interpolation=cv2.INTER_NEAREST))
    show_image_as_table(image, title="Image numérique")
    show_image_as_table(integral_image(image), title="Image intégrale")