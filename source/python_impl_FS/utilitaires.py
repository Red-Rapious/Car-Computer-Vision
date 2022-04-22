READING_MODE = "CV2" #TODO: changer pour une enum


import numpy as np # utilisation de numpy pour accéler les calculs
import RectangleRegion
import os
import cv2

def integral_image(image: list) -> list: # int array array -> int array array
    """ Convertit une image en son image intégrale """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)

    for x in range(len(image)):
        for y in range(len(image[0])):
            # relation de récurrence définie par Viola-Jones
            s[x][y] = (s[x][y-1] if y>=1 else 0) + image[x][y]
            ii[x][y] = (ii[x-1][y] if x>=1 else 0) + s[x][y]
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
    if READING_MODE == "CV2":
            return cv2.imread(path, -1)
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