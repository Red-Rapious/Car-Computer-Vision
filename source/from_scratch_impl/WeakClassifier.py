from utilitaires import evaluation

# TODO: corriger la faute de frappe "treshold" à la place de "threshold" puisré-entraîner le modèle
class WeakClassifier:
    def __init__(self, positive_regions: list, negative_regions: list, treshold: float, polarity: int):
        assert(polarity == -1 or polarity == 1)
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.treshold = treshold
        self.polarity = polarity

    def classify(self, ii: list):
        fx = evaluation(ii, self.positive_regions, self.negative_regions)
        if self.polarity * fx < self.polarity * self.treshold:
            return 1
        else:
            return 0