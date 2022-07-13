"""
| Créé par Antoine Groudiev - 2022
| - Implémentation des travaux de Paul Viola et Michael Jones
| - Sur un tutoriel original de Anmol Parande 
| - https://medium.datadriveninvestor.com/understanding-and-implementing-the-viola-jones-image-classification-algorithm-85621f7fe20b
"""

import numpy as np
import pickle

from sympy import I
from ViolaJones import ViolaJones
from CascadeClassifier import CascadeClassifier
import time

T = 1

SAVE_FOLDER = "saves/"
IMAGES_FOLDER = "././ressources/faces_images/"
PICKLE_IMAGES = IMAGES_FOLDER + "pickle_files/"

def train_viola(t):
    with open(PICKLE_IMAGES + "training.pkl", 'rb') as f:
        training = pickle.load(f)
    clf = ViolaJones(feature_number=t)
    clf.train(training, training_len=2429, test_len=4548)
    evaluate(clf, training)
    clf.save(str(t))

def test_viola(filename):
    with open(PICKLE_IMAGES + "test.pkl", 'rb') as f:
        test = pickle.load(f)
    
    clf = ViolaJones.load(filename)
    evaluate(clf, test)

def train_cascade(layers, filename="Cascade"):
    with open(PICKLE_IMAGES + "training.pkl", 'rb') as f:
        training = pickle.load(f)
    
    clf = CascadeClassifier(layers)
    clf.train(training)
    evaluate(clf, training)
    clf.save(filename)

def test_cascade(filename="Cascade"):
    with open(PICKLE_IMAGES + "test.pkl", "rb") as f:
        test = pickle.load(f)
    
    clf = CascadeClassifier.load(SAVE_FOLDER + filename)
    evaluate(clf, test)

def evaluate(clf, data):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0

    for x, y in data:
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1
        
        correct += 1 if prediction == y else 0
    
    print("[RESULTATS]")
    print("Pourcentage de Faux Positifs : %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("Pourcentage de Faux Négatifs : %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Précision : %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Temps moyen de classification : %f" % (classification_time / len(data)))

if __name__ == "__main__":
    print("[DEBUT DU PROGRAMME]")
    print("Paramètres : T = %d (nombre de features)" % T)
    print("[Entraînement du modèle] ...")
    temps_depart = time.time()
    train_viola(1)
    print("Temps d'entraînement total : %f" % (time.time() - temps_depart))
    print("[Test du modèle]")
    test_viola(SAVE_FOLDER + str(T) + ".pkl")

    print("[FIN DU PROGRAMME]")