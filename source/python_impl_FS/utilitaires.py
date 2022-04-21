import numpy as np # utilisation de numpy pour accéler les calculs
import RectangleRegion
import os

def integral_image(image: list) -> list: # a' array array -> a' array array
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

def load_images(positive_folder: str, negative_folder: str, extention:str=".pgm"):
    """ Charge les images des dossiers indiqués, et les transformes en tableaux """
    # Récupération des noms de fichiers de toutes les images de test
    positive_images = []
    for root, dirs, files in os.walk(positive_folder):
        for file in files:
            if file.endswith(extention):
                positive_images.append(os.path.join(root,file))

    negative_images = []
    for root, dirs, files in os.walk(negative_folder):
        for file in files:
            if file.endswith(extention):
                negative_images.append(os.path.join(root,file))