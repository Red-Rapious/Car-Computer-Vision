import numpy as np
import copy

class ViolaJones:
    def __init(self, feature_number):
        self.feature_number = feature_number

    def train(self, training): # (a' array array, bool) array -> void
        """
        Entraîne les classificateurs à partir des données fournies.
        training : [(ii, is_positive_example)]
        ii: image intégrale de l'image d'entraînement (a' array array)
        is_positive_example : 'True' si l'image contient l'objet cherché (bool)
        """
        training_data = copy.deepcopy(training)

        # Comptage du nombres d'exemples positifs et négatifs
        pos_count = 0
        neg_count = 0
        for (ii, is_positive_example) in training_data:
            if is_positive_example:
                pos_count += 1
            else:
                neg_count += 1
        
        # Calcul des poids
        #weights = np.array([1.0 / (2 * (pos_count if training_data[i][1] else neg_count)) for i in range(len(training_data))])
        weights = np.zeros(len(training_data))
        for i in range(len(training_data)):
            (ii, is_positive_example) = training_data[i]
            if is_positive_example:
                weights[i] = 1.0 / (2 * pos_count)
            else:
                weights[i] = 1.0 / (2 * neg_count)

    def build_features(self, image_shape):
        """
        Construit toutes les caractéristiques (=features) possibles
        à partir de tous les rectangles dans l'image de taille 'image_shape'
        """
        height, width = image_shape
        features = []

        return features