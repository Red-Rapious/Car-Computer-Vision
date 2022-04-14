import numpy as np
import copy
from RectangleRegion import RectangleRegion
from utilitaires import evalutation

class ViolaJones:
    def __init__(self, feature_number: int):
        self.feature_number = feature_number

    def train(self, training: list) -> None: # (a' array array, bool) array -> void
        """
        Entraîne les classificateurs à partir des données fournies.
        training : (ii, is_positive_example) array
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

        features = self.build_features(training_data[0][0].shape)
        X, y = self.apply_features(features, training_data)

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
                for x in range(width-w):
                    for y in range(height-h):
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
                            features.append(([right], [immediate, right_2]))
                        if y + 3 * h < height: # Adjacent horizontalement
                            features.append(([bottom], [immediate, bottom_2]))

                        # -- Features à 4 rectangles
                        features.append(([right, bottom], [immediate, bottom_right]))
        return features

    def apply_features(self, features: list, training_data: list) -> tuple:
        """ Evalue chaque feature sur chaque exemple d'entraînement """

        X = np.zeros((len(features), len(training_data)))
        y = np.array(map(lambda data: data[1], training_data)) # tableau de booléens

        i = 0
        for pos, neg in features:
            X[i] = [self.__evaluation(training_data[j][0], pos, neg) for j in range(len(training_data))]
            i += 1

        return X, y