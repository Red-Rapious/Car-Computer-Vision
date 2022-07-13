READING_MODE = "CUSTOM" #TODO: changer pour une enum


import numpy as np # utilisation de numpy pour accéler les calculs
import RectangleRegion
import os
import cv2
from random import randrange

def integral_image(image: list) -> list: # int array array -> int array array
    """ Convertit une image en son image intégrale """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)

    for y in range(len(image)):
        for x in range(len(image[y])):
            # relation de récurrence définie par Viola-Jones
            s[y][x] = (s[y-1][x] if y>=1 else 0) + image[y][x]
            ii[y][x] = (ii[y][x-1] if x>=1 else 0) + s[y][x]
    return ii

def evaluation(ii: list, positive_region: RectangleRegion, negative_region: RectangleRegion) -> int:
    """ Fonction d'évaluation de la précision d'une feature sur une image intégrale """
    score = 0
    for pos in positive_region:
        score += pos.compute_feature(ii)
    for neg in negative_region:
        score -= neg.compute_feature(ii)
    return score

def read_image(path: str) -> list: # str -> int array array
    """ Lit une image à partir d'un chemin de fichier.
    Le mode de lecture dépend de la constante READING_MODE définie plus haut """
    if READING_MODE == "CV2":
        return cv2.imread(path, -1)
    elif READING_MODE == "CUSTOM":
        with open(path, 'rb') as file:
            assert file.readline() == b"P5\n"
            (width, height) = [int(i) for i in file.readline().split()]
            depth = int(file.readline())
            assert depth <= 255

            raster = []
            for y in range(height):
                row = []
                for y in range(width):
                    row.append(ord(file.read(1)))
                raster.append(row)
            return raster
    else:
        print("[Erreur] Mode de lecture de l'image non défini")
        exit(0)

def load_images(positive_folder: str, negative_folder: str, extention:str=".pgm") -> list:
    """ Charge les images des dossiers indiqués, et les transforme en tableaux.
                str, str, str -> (int array array, bool) array """
    training_data = []
    
    # Récupération des noms de fichiers de toutes les images de test
    positive_images_paths = []
    for root, dirs, files in os.walk(positive_folder):
        for file in files:
            if file.endswith(extention):
                positive_images_paths.append(os.path.join(root,file))

    negative_images_paths = []
    for root, dirs, files in os.walk(negative_folder):
        for file in files:
            if file.endswith(extention):
                negative_images_paths.append(os.path.join(root,file))

    # Ajout de chaque image sous forme d'un tableau de nombres
    for path in positive_images_paths:
        training_data.append((read_image(path), True))

    for path in negative_images_paths:
        training_data.append((read_image(path), False))

    return training_data


if __name__ == "__main__":
    images = load_images("ressources/training_data/train/face", "ressources/training_data/train/non-face")
    gray_image = cv2.cvtColor(np.array(images[randrange(len(images))][0]).astype('uint8'), cv2.COLOR_GRAY2BGR)

    gray_image = cv2.resize(gray_image, (0,0), fx=5, fy=5)

    cv2.imshow("image", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()