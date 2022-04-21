from numpy import negative
from ViolaJones import ViolaJones
import pickle

class CascadeClassifier:
    def __init__(self, layers:list):
        """ 'layers' est un tableau contenant le nombre de features pour chaque layer """
        self.layers = layers
        self.classifiers = []

    def train(self, training: list):
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
        for clf in self.classifiers:
            if not clf.classify(image):
                return False
        return True

    def save(self, filename=str) -> None:
        """ Utilise le module Pickle pour sauvegarder le modèle entraîné"""
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(self, filename=str):
        """ Utilise le module Pickle pour charger un modèle enregistré """
        with open(filename + ".pkl", 'r') as f:
            return pickle.load(f)