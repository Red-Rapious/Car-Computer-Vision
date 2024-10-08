""" miscellaneous.py - Utilitaires externes
Fichier comprenant des fonctions annexes diverses, qui ne servent pas directement 
au fonctionnement de l'algorithme mais qui génèrent des images pour le diaporama.
"""


from ViolaJones import ViolaJones
from WeakClassifier import WeakClassifier
from RectangleRegion import RectangleRegion
from utilitaires import integral_image, read_image

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas
import glob


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
    """ Remplit un rectangle par une couleur unie, noir ou blanc, sur une image """

    col = (220, 220, 220) if is_positive else (0, 0, 0) # blanc si positif, noir si négatif
    for x in range(angle1[0], angle2[0]):
        for y in range(angle1[1], angle2[1]):
            image[y][x] = col # Remplit le pixel de la couleur désirée

def draw_weakclassifier_on_image(image: list, weak_classifier: WeakClassifier) -> None:
    """ Dessine la feature d'un WeakClassifier sur une image """
    is_polarity_pos = True if weak_classifier.polarity == 1 else False

    # Remplit les régions positives en blanc et les régions négatives en noir, 
    # et inversement pour une polarité négative
    for region in weak_classifier.positive_regions:
        fill_rectangle(image, (region.x, region.y), (region.x + region.width, region.y + region.height), is_polarity_pos)
    for region in weak_classifier.negative_regions:
        fill_rectangle(image, (region.x, region.y), (region.x + region.width, region.y + region.height), not is_polarity_pos)

def draw_main_classifiers_on_image(image: list, classifiers: list) -> None:
    """ Dessine les features d'une liste de WeakClassifier sur une image """
    for classifier in classifiers:
        draw_weakclassifier_on_image(image, classifier)

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

def show_image_as_table(image: list, title: str) -> None:
    """ Affiche une image sous forme de tableau de nombres """
    # Définition de la figure et des axes
    fig, ax = plt.subplots(num=title)

    # Suppression des axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    # Conversion de l'image en DataFrame de la bibliothèque Pandas
    df = pandas.DataFrame(image)

    # Création du tableau
    _ = ax.table(cellText=df.values, loc='center')

    # Affichage de la figure
    fig.tight_layout()
    plt.show()

def show_integral_image_process(file_path:str = "ressources/training_images/faces_images/train/face/face00324.pgm") -> None:
    image = np.array(read_image(file_path), dtype=np.uint8)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Affichage de l'image en nuances de gris
    cv2.imshow("Image en nuances de gris", cv2.resize(image, (0, 0), fx=30, fy=30, interpolation=cv2.INTER_NEAREST))
    # Affichage de l'image sous forme de tableau de nombres
    show_image_as_table(image, title="Image numérique")
    # Affichage de l'image intégrale sous forme de tableau de nombres
    show_image_as_table(np.int_(integral_image(image)), title="Image intégrale")

def display_data_sample_table(folder_path: str, nb_images_side: tuple, resolution: tuple):
    folder = glob.glob(folder_path)

    data = []
    for image_path in folder:
        data.append(read_image(image_path))
        if len(data) >= nb_images_side[0] * nb_images_side[1]:
            break

    full_image_table = [[0 for _ in range(nb_images_side[0] * resolution[0])] for _ in range(nb_images_side[1] * resolution[1])]
    
    for i in range(nb_images_side[0]):
        for j in range(nb_images_side[1]):
            if i * nb_images_side[1] + j >= len(data):
                print("Pas assez d'images")
                break
            image = data[i * nb_images_side[1] + j]
            for x in range(len(image)):
                for y in range(len(image[0])):
                    full_image_table[j*resolution[1] + y][i*resolution[0] + x] = image[y][x]

    full_image_table = np.array(full_image_table)
    full_image_table = cv2.resize(full_image_table, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Training images", full_image_table)
    cv2.waitKey(0)

def classifier_superposition_face_example():
    clf = ViolaJones(feature_number=5)
    clf.classifiers = [WeakClassifier([RectangleRegion(9, 4, 2, 6)], [RectangleRegion(7, 4, 2, 6)], 0.0, 1),
                       WeakClassifier([RectangleRegion(8, 1, 3, 5)], [RectangleRegion(2, 1, 6, 5), RectangleRegion(11, 1, 6, 5)], 0.0, 1),
                       WeakClassifier([RectangleRegion(7, 17, 5, 2)], [RectangleRegion(7, 15, 5, 2)], 0.0, 1),
                       WeakClassifier([RectangleRegion(13, 10, 5, 3)], [RectangleRegion(13, 13, 5, 3)], 0.0, 1),
                       WeakClassifier([RectangleRegion(1, 11, 5, 4)], [RectangleRegion(1, 15, 5, 4)], 0.0, 1)]

    image = cv2.cvtColor(np.array(read_image("ressources/training_images/face_images/train/face/face00324.pgm"), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    draw_main_classifiers_on_image(image, clf.classifiers)

    image = cv2.resize(image, (0, 0), fx=15, fy=15, interpolation=cv2.INTER_NEAREST)

    cv2.imshow(" ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    '''
    clf = ViolaJones.load("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/from_scratch_impl/saves/stop_sign_v2_cascade_1_5_10/sub_stop_sign_v2_10")

    image = cv2.cvtColor(np.array(read_image("ressources/training_images/stop_sign_images/stop_signs_images_processed/train/stop_sign_train_44.pgm"), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    #draw_main_classifiers_on_image(image, clf.classifiers)

    image = cv2.resize(image, (0, 0), fx=15, fy=15, interpolation=cv2.INTER_NEAREST)

    cv2.imshow(" ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    show_integral_image_process("ressources/training_images/stop_sign_images/stop_signs_images_processed/train/stop_sign_train_44.pgm")
    '''

    #display_data_sample_table("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_sign_v2_images/test_variance_stop_sign_v2_images_processed/train/*", (16, 16), (19, 19))

    """
    image = cv2.cvtColor(np.array(read_image("ressources/training_images/stop_sign_images/stop_signs_images_processed/train/stop_sign_train_44.pgm"), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (0, 0), fx=500/19, fy=500/19, interpolation=cv2.INTER_NEAREST)
    fill_rectangle(image, (0, 0), (499, 499), True)
    fill_rectangle(image, (10, 10), (249, 249), False)
    fill_rectangle(image, (249, 10), (489, 249), False)
    cv2.imshow(" ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    """
    #classifier_superposition_face_example()
    image = cv2.cvtColor(np.array(read_image("ressources/training_images/face_images/train/non-face/B20_03430.pgm"), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    #draw_main_classifiers_on_image(image, clf.classifiers)

    image = cv2.resize(image, (0, 0), fx=15, fy=15, interpolation=cv2.INTER_NEAREST)
    """

    tab = np.array(read_image("ressources/training_images/stop_sign_images/stop_signs_images_processed/train/stop_sign_train_44.pgm"), dtype=np.uint8)
    tab = integral_image(tab)
    tab = tab * 255 / tab[18][18]
    tab = np.round(tab).astype(int)
    grayscale = np.array(tab, dtype=np.uint8)
    image = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (0, 0), fx=500/19, fy=500/19, interpolation=cv2.INTER_NEAREST)    
    cv2.imshow(" ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()