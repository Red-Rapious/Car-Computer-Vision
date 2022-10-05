'''
    Fichier comprenant des fonctions annexes diverses, qui ne servent pas directement 
    au fonctionnement de l'algorithme mais qui génèrent des images pour le diaporama.
'''


from ViolaJones import ViolaJones
from WeakClassifier import WeakClassifier
from RectangleRegion import RectangleRegion
from utilitaires import integral_image, read_image

import matplotlib.pyplot as plt
import numpy as np
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

def draw_random_classifier_on_video() -> None:
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

def show_integral_image_process(filepath:str = "ressources/training_images/faces_images/train/face/face00324.pgm") -> None:
    image = np.array(read_image(filepath), dtype=np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Image en nuances de gris", cv2.resize(image, (0, 0), fx=30, fy=30, interpolation=cv2.INTER_NEAREST))
    show_image_as_table(image, title="Image numérique")
    show_image_as_table(np.int_(integral_image(image)), title="Image intégrale")

def draw_main_classifiers_on_image(image: list, classifiers: list) -> None:
    for classifier in classifiers:
        draw_weakclassifier_on_image(image, classifier)

if __name__ == "__main__":
    clf = ViolaJones.load("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/from_scratch_impl/saves/stop_sign_v2_cascade_1_5_10/sub_stop_sign_v2_10")

    image = cv2.cvtColor(np.array(read_image("ressources/training_images/stop_sign_images/stop_signs_images_processed/train/stop_sign_train_44.pgm"), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    #draw_main_classifiers_on_image(image, clf.classifiers)

    image = cv2.resize(image, (0, 0), fx=15, fy=15, interpolation=cv2.INTER_NEAREST)

    cv2.imshow(" ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    show_integral_image_process("ressources/training_images/stop_sign_images/stop_signs_images_processed/train/stop_sign_train_44.pgm")