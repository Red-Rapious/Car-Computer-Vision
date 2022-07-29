'''
    Fichier comprenant des fonctions directement utiles aux différents fichiers de l'algorithme.
'''

import numpy as np # utilisation de numpy pour accéler les calculs
import RectangleRegion
import os
import cv2
import enum
import pickle

class ReadingMode (enum.Enum):
    CV2 = 0
    CUSTOM = 1

class AccuracyMethod (enum.Enum):
    STANDARD = 0
    FSCORE = 1

READING_MODE = ReadingMode.CUSTOM

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
    if READING_MODE == ReadingMode.CV2:
        return cv2.imread(path, -1)
    elif READING_MODE == ReadingMode.CUSTOM:
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

def load_images(positive_folder: str, negative_folder: str, extention:str=".pgm", max_negatives:int=-1) -> list:
    """ Charge les images des dossiers indiqués, et les transforme en tableaux.
                str, str, str -> (int array array, bool) array """
    training_data = []
    
    # Récupération des noms de fichiers de toutes les images de test
    pos_count = 0
    positive_images_paths = []
    for root, dirs, files in os.walk(positive_folder):
        for file in files:
            if file.endswith(extention):
                positive_images_paths.append(os.path.join(root,file))
                pos_count += 1

    neg_count = 0
    negative_images_paths = []
    for root, dirs, files in os.walk(negative_folder):
        for file in files:
            if file.endswith(extention):
                if max_negatives != -1 and neg_count >= max_negatives:
                    break
                negative_images_paths.append(os.path.join(root,file))
                neg_count += 1

    # Ajout de chaque image sous forme d'un tableau de nombres
    for path in positive_images_paths:
        training_data.append((np.array(read_image(path)), True))

    for path in negative_images_paths:
        training_data.append((np.array(read_image(path)), False))

    print("[INFO] Chargement des images terminé")
    print("     {} images positives".format(pos_count))
    print("     {} images negatives".format(neg_count))

    return training_data

def images_to_pickle(name: str, positive_folder: str, negative_folder: str, extention:str=".pgm", max_negatives:int=-1):
    """ Sauvegarde les images des dossiers indiqués dans un fichier pickle """
    training_data = load_images(positive_folder, negative_folder, extention, max_negatives=max_negatives)
    with open(name + ".pkl", 'wb') as file:
        pickle.dump(training_data, file)

def measure_accuracy(true_positives:int, true_negatives: int, false_positives: int, false_negatives: int, method:AccuracyMethod = AccuracyMethod.STANDARD) -> int:
    if method == AccuracyMethod.STANDARD:
        tot_true = true_positives + true_negatives
        tot_data = true_positives + true_negatives + false_negatives + false_positives
        return tot_true / tot_data
    if method == AccuracyMethod.FSCORE:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        return 2 / (1/precision + 1/recall)

if __name__ == "__main__":
    print("     [DEBUT DU PROGRAMME]")
    name = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_sign_images/pickle_files/test"
    positive_folder = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/stop_sign_images/stop_signs_images_processed/test"
    negative_folder = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/training_images/face_images/test/non-face"
    images_to_pickle(name, positive_folder, negative_folder, ".pgm", max_negatives=56*50)
    print("     [FIN DU PROGRAMME]")