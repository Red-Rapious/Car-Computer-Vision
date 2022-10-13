""" image_pickler.py - Utilitaire externe
Transforme des images en une liste contenant l'image sous forme de tableau et sa classification.
"""

import pickle
import glob
from utilitaires import read_image

def save(data, filename:str) -> None:
        """ Utilise le module Pickle pour sauvegarder le modèle entraîné"""
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    data = []
    pos_folder = glob.glob("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/fullsize_test_images/positives/*.jpg")
    neg_folder = glob.glob("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/fullsize_test_images/negatives/*.jpg")
    neg_folder2 = glob.glob("/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/fullsize_test_images/negatives/kaggle/*.jpg")

    
    for image_path in pos_folder+neg_folder+neg_folder2:
        data.append((read_image(image_path), 1 if image_path in pos_folder else 0))
        
    save(data, "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/ressources/fullsize_test_images/pickle_files/fullsize_test")