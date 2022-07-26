from utilitaires import evaluation

class WeakClassifier:
    def __init__(self, positive_regions: list, negative_regions: list, treshold: float, polarity: int):
        assert(polarity == -1 or polarity == 1)
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.treshold = treshold
        self.polarity = polarity

    def classify(self, ii: list):
        return 1 if self.polarity * evaluation(ii, self.positive_regions, self.negative_regions) < self.polarity * self.treshold else 0
