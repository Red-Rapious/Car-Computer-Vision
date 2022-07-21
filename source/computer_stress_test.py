import numpy as np
from random import randrange, seed
import time

# CONSTANTES
# N = 10 # nombre de matrices à inverser
# T = 1000 # taille des matrices
# M = 100 # valeur max des éléments

def stress_test(N: int, T: int, M: int, number: str):
    start = time.time()
    print("[Début du test " + number + "]")

    s = 0 # seed fixée pour avoir toujours les mêmes résultats
    for i in range(N):
        s += 12345
        seed(s)

        # Création d'une matrice de taille TxT
        A = np.array([[randrange(1, M+1) for j in range(T)] for i in range(T)])
        det = np.linalg.det(A)
        if det != 0:
            # Inversion de la matrice
            B = np.linalg.inv(A)
        print("     ["+str(i+1)+"/"+str(N)+"]")

    print("[Info] Temps total du test", number, ":", round(time.time() - start, 2), "secondes")
    print("[Fin du test " + number + "]\n")
    return round(time.time() - start, 2)

def total_score():
    score = 0
    score += stress_test(10, 500, 1000, "1")
    score += stress_test(8, 1000, 1000, "2")
    score += stress_test(6, 2000, 1000, "3")
    score += stress_test(4, 3000, 1000, "4")
    score += stress_test(2, 4000, 1000, "5")
    return score

if __name__ == "__main__":
    print("[DEBUT DU PROGRAMME")
    score_total = total_score()
    print("[INFO] Score total :", score_total)
    print("[FIN DU PROGRAMME]")