"""
| Créé par Antoine Groudiev - 2022
| - Implémentation des travaux de Paul Viola et Michael Jones
| - Sur un tutoriel original de Anmol Parande 
| - https://medium.datadriveninvestor.com/understanding-and-implementing-the-viola-jones-image-classification-algorithm-85621f7fe20b
"""

import numpy as np
import pickle

from ViolaJones import ViolaJones
from CascadeClassifier import CascadeClassifier
import time

from utilitaires import AccuracyMethod, measure_accuracy

# HYPERPARAMÈTRES
T = 5 # nombre de classificateurs faibles

# DEBUG
IMG_NUMBER = 3
TRAIN_MODEL = False
TEST_MODEL = True

SAVE_FOLDER = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/from_scratch_impl/saves/"
IMAGES_FOLDER = "././ressources/training_images/faces_images/"
PICKLE_IMAGES = IMAGES_FOLDER + "pickle_files/"
OBJECT = "face"

def train_viola(t):
    with open(PICKLE_IMAGES + "training.pkl", 'rb') as f:
        training = pickle.load(f)
    clf = ViolaJones(feature_number=t)
    clf.train(training[:IMG_NUMBER], training_len=2429, test_len=4548)
    evaluate(clf, training)
    clf.save(SAVE_FOLDER + OBJECT + str(t))

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
    tot_negatives, tot_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0

    for x, y in data:
        if y == 1:
            tot_positives += 1
        else:
            tot_negatives += 1

        start = time.time()
        prediction = clf.classify(x)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1
        if prediction == 0 and y == 0:
            true_negatives += 1
        if prediction == 1 and y == 1:
            true_positives += 1
        
        correct += 1 if prediction == y else 0
    
    print("\n[RESULTATS]")
    print(" Pourcentage de Faux Positifs : %d/%d (%f)" % (false_positives, tot_negatives, false_positives/tot_negatives))
    print(" Pourcentage de Faux Négatifs : %d/%d (%f)" % (false_negatives, tot_positives, false_negatives/tot_positives))
    standard = measure_accuracy(true_positives, true_negatives, false_positives, false_negatives, AccuracyMethod.STANDARD)
    fscore = measure_accuracy(true_positives, true_negatives, false_positives, false_negatives, AccuracyMethod.FSCORE)

    #print(" Accuracy : %f" % accuracy)
    print("'Précision' (accuracy) :")
    print("     Méthode Standard : %d/%d (%f)", (correct, len(data), correct/len(data)))
    print("     F-Score : ", fscore)
    print(" Temps moyen de classification : %fs" % (classification_time / len(data)))

if __name__ == "__main__":
    print("\n\n     --- [DEBUT DU PROGRAMME] ---\n")
    print("[Paramètres] : ")
    print("     T = %d (nombre de classificateurs faibles)" % T)

    if TRAIN_MODEL:
        print("     [DEBUG] Le paramètre 'IMG_NUMBER' est activé - toute la base d'images ne sera pas utilisée. \n         Définir IMG_NUMBER = -1 pour utiliser toute la base d'images.\n     IMG_NUMBER = ", IMG_NUMBER) if IMG_NUMBER != -1 else None
        print("\n[Entraînement du modèle] ...")
        temps_depart = time.time()
        train_viola(T)
        print(" Temps d'entraînement total : %f min" % (round(((time.time() - temps_depart)/60), 0)))
    if TEST_MODEL:
        print("\n[Test du modèle]")
        test_viola(SAVE_FOLDER + OBJECT + str(T))

    print("\n       --- [FIN DU PROGRAMME] ---\n\n")