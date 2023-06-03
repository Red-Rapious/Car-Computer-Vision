""" ViolaJones.py - Classe interne
"""

import pickle
import numpy as np
import copy
from RectangleRegion import RectangleRegion
from WeakClassifier import WeakClassifier
from utilitaires import evaluation, integral_image
import time

from sklearn.feature_selection import SelectPercentile, f_classif

class ViolaJones:
    def __init__(self, feature_number: int):
        self.feature_number = feature_number
        self.alphas = []
        self.classifiers = []

    def train(self, training: list, percentile:int=10) -> None:
        """
        Entraîne les classificateurs à partir des données fournies.
        training: (ii, is_positive_example) array
            ii: image intégrale de l'image d'entraînement (a' array array)
            is_positive_example : 'True' si l'image contient l'objet cherché (bool)
        percentile: pourcentage du nombre de features à garder avec SciKit-Learn
        """

        # Comptage du nombres d'exemples positifs et négatifs
        print("Comptage des exemples...")
        pos_count = 0
        neg_count = 0
        for (ii, is_positive_example) in training:
            if is_positive_example:
                pos_count += 1
            else:
                neg_count += 1
        
        # Calcul des poids
        print("Initialisation des poids et transformation des images en images intégrales...")
        #weights = np.array([1.0 / (2 * (pos_count if training_data[i][1] else neg_count)) for i in range(len(training_data))])
        training_data = []
        weights = np.zeros(len(training))
        for i in range(len(training)):
            training_data.append((integral_image(training[i][0]), training[i][1]))
            if training[i][1]:
                weights[i] = 1.0 / (2 * pos_count)
            else:
                weights[i] = 1.0 / (2 * neg_count)

        print("Création des features...")
        features = self.build_features(training_data[0][0].shape)
        print(str(len(features)) + " features créées.")
        print("Application des features...")
        X, y = self.apply_features(features, training_data)
        
        # Utilisation du module SciKit-Learn pour choisir les features les plus importantes
        print("Sélection des meilleures features (SciKit)...")
        indices = SelectPercentile(f_classif, percentile=percentile).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        print(str(len(features)) + " features conservées.")

        print("Mise à jour des poids...")
        for t in range(self.feature_number):
            print("Démarrage de la sélection des classificateurs pour la feature " + str(t+1) + "/" + str(self.feature_number) +  "...")
            
            weights = weights / np.linalg.norm(weights) # normalisation des poids
            weak_classifiers = self.train_weak_classifiers(X, y, features, weights)

            # sélection du meilleur classificateur
            clf, error, accuracy = self.select_best_classifier(weak_classifiers, weights, training_data)
            self.classifiers.append(clf)

            # mise à jour des poids
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = -np.log(beta)
            self.alphas.append(alpha)
            print("Classificateur %s choisi, avec une précision %f et un alpha %f \n" % (str(clf), len(accuracy) - sum(accuracy), alpha))

        print("Phase d'entraînement terminée.")
        


    def build_features(self, image_shape: tuple) -> list:
        """
        Construit toutes les caractéristiques (=features) possibles
        à partir de tous les rectangles dans l'image de taille 'image_shape'
        image_shape: (int, int), dimensions de l'image

        -> features : (RectangleRegion array, RectangleRegion array) array
        Chaque tuple contient les régions positives et les régions négatives de la feature
        """
        height, width = image_shape
        features = []

        for w in range(1, width+1):
            for h in range(1, height+1):
                # Pour toutes les dimensions possibles de rectangles
                x = 0
                while x + w < width:
                    y = 0
                    while y + h < height:
                        # Pour tous les rectangles possibles
                        
                        # -- Création des différentes sous-zones possibles
                        immediate = RectangleRegion(x, y, w, h)
                        right = RectangleRegion(x + w, y, w, h)
                        bottom = RectangleRegion(x, y + h, w, h)

                        right_2 = RectangleRegion(x + 2*w, y, w, h)
                        bottom_2 = RectangleRegion(x, y + 2*h, w, h)

                        bottom_right = RectangleRegion(x + w, y + h, w, h)

                        # -- Features à 2 rectangles
                        if x + 2*w < width: # Adjacent horizontalement
                            features.append(([right], [immediate]))
                        if y + 2*h < height: # Adjacent verticalement
                            features.append(([immediate], [bottom]))

                        # -- Features à 3 rectangles
                        if x + 3 * w < width: # Adjacent horizontalement
                            features.append(([right], [right_2, immediate]))
                        if y + 3 * h < height: # Adjacent horizontalement
                            features.append(([bottom], [bottom_2, immediate]))

                        # -- Features à 4 rectangles
                        if x + 2 * w < width and y + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        y += 1
                    x += 1
        return np.array(features, dtype=object)

    def apply_features(self, features: list, training_data: list) -> tuple:
        """ Évalue chaque feature sur chaque exemple d'entraînement """

        X = np.zeros((len(features), len(training_data)))
        y = list(map(lambda data: data[1], training_data)) # tableau de booléens

        i = 0
        modulo = 10000
        last_time = time.time()
        for pos, neg in features:
            # Message de progression
            if i%modulo == 0 and i != 0:
                remaining_time = round((time.time() - last_time) * (len(features) - i) / modulo / 60, 0)
                print("     [INFO] Avancée :", str(i) + "/" + str(len(features)), "     Temps pour", modulo, "features : " + str(round(time.time() - last_time, 2)) + "s    Temps restant estimé : " + str(remaining_time) + "min")
                last_time = time.time()
            
            X[i] = [evaluation(training_data[j][0], pos, neg) for j in range(len(training_data))]
            i += 1
        print("\n")

        return X, y

    def train_weak_classifiers(self, X: list, y: list, features: list, weight: list) -> list:
        """ Entraîne les Weak Classifiers à l'aide de la méthode de calcul de l'erreur minimale"""

        # Compte le nombre de classificateurs positifs et négatifs
        total_pos, total_neg = 0, 0
        for w, is_positive in zip(weight, y):
            if is_positive:
                total_pos += w
            else:
                total_neg += w
        
        classifiers = []
        total_features = len(X)
        for index, feature in enumerate(X):
            # Affichage de la progression du classement, qui peut être long
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                percentage = str(int(100 * len(classifiers) / total_features))
                print("[Progression] :", len(classifiers), "sur", total_features, "(" + percentage + " %)")
            
            # Recherche du classifier avec la plus petite erreur
            applied_feature = sorted(zip(weight, feature, y), key= lambda x: x[1])
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float("inf"), None, None, None

            for w, f, is_positive in applied_feature:
                # Calcul de l'erreur
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                
                # Mise à jour des éléments avec erreur minimale
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if is_positive:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            # Création et ajout du classificateur optimal
            if best_feature is not None:
                classifier = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
                classifiers.append(classifier)
            else:
                pass
                #print("[WARNING] Aucun classificateur optimal n'a été trouvé pour la feature", index)
                #print("[DEBUG INFOS] len(applied_feature) :", len(applied_feature))
                # Erreurs valent NaN
        
        return classifiers

    def select_best_classifier(self, classifiers: list, weights: list, training_data: list) -> list:
        """ Choisit le meilleur classificateur """
        
        best_clf, best_error, best_accuracy = None, float("inf"), None
        
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error/len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        
        # note : pour éviter d'avoir une erreur strictement nulle qui nuirait à la suite des calculs,
        # on met à la place une erreur minuscule
        return best_clf, max(best_error, 0.001), best_accuracy

    def classify(self, image: list, alreadyII: bool=False) -> bool:
        """ Indique si une image contient ou non l'objet précédemment classifié """
        total = 0
        if alreadyII:
            ii = image
        else:
            ii = integral_image(image)
        
        for (alpha, clf) in zip(self.alphas, self.classifiers):
            total += alpha * clf.classify(ii)
        
        # TODO: RETABLIR LA VALEUR INITIALE DE 0.5
        return total >= 0.5 * sum(self.alphas)

    def save(self, filename:str) -> None:
        """ Utilise le module Pickle pour sauvegarder le modèle entraîné"""
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename:str):
        """ Utilise le module Pickle pour charger un modèle enregistré """
        with open(filename + ".pkl", 'rb') as f:
            return pickle.load(f)