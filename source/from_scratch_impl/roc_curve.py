import numpy as np
import pickle
import time
from random import shuffle, seed
import matplotlib.pyplot as plt

from ViolaJones import ViolaJones
from CascadeClassifier import CascadeClassifier
from utilitaires import AccuracyMethod, measure_accuracy
from full_image_detection import apply_cascade_to_image

FULLSIZE = True

# HYPERPARAMÈTRES
OBJECT = "stop_sign_v2"
CASCADE_NAME = "cascade_1_5_10"
CASCADE_LAYERS = [1, 5, 10, 25, 50, 75, 100]

SAVE_FOLDER = "/Users/antoinegroudiev/Documents/Code/Car-Computer-Vision/source/from_scratch_impl/saves/"
IMAGES_FOLDER = "././ressources/training_images/" + OBJECT + "_images/"
PICKLE_IMAGES = IMAGES_FOLDER + "pickle_files/"
seed(12345)

def load_training_data():
    with open(PICKLE_IMAGES + "train.pkl", 'rb') as f:
        training = pickle.load(f)
    shuffle(training)
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

def load_fullsize_data():
    with open("././ressources/fullsize_test_images/pickle_files/fullsize_test.pkl", 'rb') as f:
        data = pickle.load(f)
    return data

def test_fullsize_cascade(filename="Cascade"):
    data = load_fullsize_data()
    shuffle(data)
    clf = CascadeClassifier.load(SAVE_FOLDER + OBJECT + "_" + filename)
    evaluate(clf, data)

def classify(x, clf):
    if FULLSIZE:
        return len(apply_cascade_to_image(x, clf)) > 0
    else:
        return clf.classify(x)

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
        prediction = classify(x, clf)
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
    
    #fig, axes = plt.subplots(figsize =(7, 5), num="Courbe ROC")
    fig, axes = plt.subplots(num="Courbe ROC")

    axes.plot([i+1 for i in range(len(curve))], curve)
    plt.xticks(range(1, len(curve)+1))
    plt.title("Courbe ROC")
    plt.ylabel("Exactitude", fontweight="bold")
    plt.xlabel("Faux Positifs", fontweight="bold")
    axes.yaxis.set_view_interval(0, 1)
    plt.show()

if __name__ == "__main__":
    print("\n\n     --- [DEBUT DU PROGRAMME] ---\n")

    test_cascade(CASCADE_NAME)

    print("\n       --- [FIN DU PROGRAMME] ---\n")