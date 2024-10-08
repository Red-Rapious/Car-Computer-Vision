""" CascadeClassifier.py - Classe interne
"""

from ViolaJones import ViolaJones
import pickle
import os
from utilitaires import integral_image

class CascadeClassifier:
    def __init__(self, layers:list):
        """ 'layers' est un tableau contenant le nombre de features T pour chaque sous-classificateur fort """
        self.layers = layers
        self.classifiers = []

    def train(self, training: list):
        """
        Entraîne les classificateurs à partir des données fournies.
        training: (ii, is_positive_example) array
            ii: image intégrale de l'image d'entraînement (a' array array)
            is_positive_example : 'True' si l'image contient l'objet cherché (bool)
        """

        pos, neg = [], []

        for example in training:
            if example[1]:
                pos.append(example)
            else:
                neg.append(example)

        for feature_num in self.layers:
            if len(neg) == 0:
                print("[INFO] Arrêt anticipé : il n'y a plus de faux positifs")
                break
            
            classifier = ViolaJones(feature_number=feature_num)
            classifier.train(pos + neg)
            self.classifiers.append(classifier)

            false_positives = []
            for example in neg:
                if self.classify(example[0]): # si l'exemple est mal classifié
                    false_positives.append(example)
            neg = false_positives

    def classify(self, image:list) -> bool:
        """ Prédit si une image contient ou non l'objet précédemment classifié """
        ii = integral_image(image)
        for clf in self.classifiers:
            if not clf.classify(ii, alreadyII=True):
                return False
        return True

    def save(self, filename:str, object:str) -> None:
        """ Utilise le module Pickle pour sauvegarder le modèle entraîné"""
        os.mkdir(filename)

        with open(filename + "/cascade.pkl", "wb") as f:
            pickle.dump(self, f)
        
        for i, clf in enumerate(self.classifiers):
            clf.save(filename + "/sub_" + object + "_" + str(self.layers[i]))

    @staticmethod
    def load(filename:str):
        """ Utilise le module Pickle pour charger un modèle enregistré """
        with open(filename + "/cascade.pkl", 'rb') as f:
            return pickle.load(f)