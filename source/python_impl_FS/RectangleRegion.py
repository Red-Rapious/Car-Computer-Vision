class RectangleRegion:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii): # a' array array -> int
        """ Calcule la valeur de l'image intégrale 
        dans la zone de cette région rectangulaire """
        p00 = ii[self.x][self.y]
        p10 = ii[self.x + self.width][self.y]
        p01 = ii[self.x][self.y + self.height]
        p11 = ii[self.x + self.width][self.y + self.height]

        return p00 + p10 - p01 - p11