from numpy import negative
from ViolaJones import ViolaJones
import pickle
import os

class CascadeClassifier:
    def __init__(self, layers:list):
        """ 'layers' est un tableau contenant le nombre de features pour chaque layer """
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
            classifier.train(pos + neg, len(pos), len(neg))
            self.classifiers.append(classifier)

            false_positives = []
            for example in neg:
                if self.classify(example[0]): # si l'exemple est mal classifié
                    false_positives.append(example)
            neg = false_positives

    def classify(self, image:list):
        """ Indique si une image contient ou non l'objet précédement classifié """
        for clf in self.classifiers:
            if not clf.classify(image):
                return False
        return True

    def save(self, filename:str) -> None:
        """ Utilise le module Pickle pour sauvegarder le modèle entraîné"""
        os.mkdir(filename)

        with open(filename + "/cascade.pkl", "wb") as f:
            pickle.dump(self, f)
        
        for i, clf in enumerate(self.classifiers):
            clf.save(filename + "/sub_face" + str(self.layers[i]))

    @staticmethod
    def load(filename:str):
        """ Utilise le module Pickle pour charger un modèle enregistré """
        with open(filename + ".pkl", 'rb') as f:
            return pickle.load(f)