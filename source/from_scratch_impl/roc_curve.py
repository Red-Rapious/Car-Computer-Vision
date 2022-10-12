import numpy as np
import pickle
import time
from random import shuffle, seed
import matplotlib.pyplot as plt

from ViolaJones import ViolaJones
from CascadeClassifier import CascadeClassifier
from utilitaires import AccuracyMethod, measure_accuracy

# HYPERPARAMÈTRES
T = 5 # nombre de classificateurs faibles en cas de modèle seul
OBJECT = "stop_sign_v2"
CASCADE_NAME = "cascade_1_5_10"
CASCADE_LAYERS = [1, 5, 10, 25, 50, 75, 100]

# DEBUG
IMG_NUMBER = -1 # nombre d'images à utiliser pour l'entraînement ; -1 pour tout le set
SUFFLE = True

SAVE_FOLDER = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/from_scratch_impl/saves/"
IMAGES_FOLDER = "././ressources/training_images/" + OBJECT + "_images/"
PICKLE_IMAGES = IMAGES_FOLDER + "pickle_files/"
seed(12345)

def load_training_data():
    with open(PICKLE_IMAGES + "train.pkl", 'rb') as f:
        training = pickle.load(f)
    if SUFFLE:
        shuffle(training)
    if IMG_NUMBER != -1:
        training = training[:IMG_NUMBER]
    return training

def load_test_data():
    with open(PICKLE_IMAGES + "test.pkl", 'rb') as f:
        test = pickle.load(f)
    return test

def test_cascade(filename="Cascade"):
    test = load_test_data()
    train = load_training_data()
    data = test + train
    clf = CascadeClassifier.load(SAVE_FOLDER + OBJECT + "_" + filename)
    evaluate(clf, data)

def evaluate(clf, data):
    correct = 0
    tot_negatives, tot_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0

    curve = []

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
            curve.append(measure_accuracy(true_positives, true_negatives, false_positives, false_negatives, AccuracyMethod.STANDARD))
        if prediction == 0 and y == 1:
            false_negatives += 1
        if prediction == 0 and y == 0:
            true_negatives += 1
        if prediction == 1 and y == 1:
            true_positives += 1
        
        correct += 1 if prediction == y else 0
    
    print("\n[RESULTATS]")
    print("Classification :")
    print("     Vrais Positifs : %d/%d (%f)" % (true_positives, tot_positives, true_positives/tot_positives))
    print("     Vrais Négatifs : %d/%d (%f)" % (true_negatives, tot_negatives, true_negatives/tot_negatives))
    print("     Faux Positifs  : %d/%d (%f)" % (false_positives, tot_negatives, false_positives/tot_negatives))
    print("     Faux Négatifs  : %d/%d (%f)" % (false_negatives, tot_positives, false_negatives/tot_positives))
    standard = measure_accuracy(true_positives, true_negatives, false_positives, false_negatives, AccuracyMethod.STANDARD)
    fscore = measure_accuracy(true_positives, true_negatives, false_positives, false_negatives, AccuracyMethod.FSCORE)

    print("'Précision' (accuracy) :")
    print("     Méthode Standard :", str(round(standard, 3)), " (%d/%d)" % (correct, len(data)))
    print("     F-Score          :", round(fscore, 3))
    print(" Temps moyen de classification :", str(classification_time / len(data)) + "s")

    plt.plot([i for i in range(len(curve))], curve)
    plt.show()

if __name__ == "__main__":
    print("\n\n     --- [DEBUT DU PROGRAMME] ---\n")

    test_cascade(CASCADE_NAME)

    print("\n       --- [FIN DU PROGRAMME] ---\n")