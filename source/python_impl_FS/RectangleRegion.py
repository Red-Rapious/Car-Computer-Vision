from numpy import int64

class RectangleRegion:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii: list) -> int: # a' array array -> int
        """ Calcule la valeur de l'image intégrale 
        dans la zone de cette région rectangulaire """
        
        try:
            p00 = int64(ii[self.y][self.x])
            p10 = int64(ii[self.y + self.height][self.x])
            p01 = int64(ii[self.y][self.x + self.width])
            p11 = int64(ii[self.y + self.height][self.x + self.width])
        except IndexError:
            print("\n\nERREUR: IndexError")
            print("x: " + str(self.x))
            print("y: " + str(self.y))
            print("width: " + str(self.width))
            print("height: " + str(self.height))
            print("ii (shape):", ii.shape)
            print("FIN DU PROGRAMME ANTICIPEE")
            exit()

        return p11 + p00 - p10 - p01