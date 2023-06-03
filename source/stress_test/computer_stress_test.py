import numpy as np
from random import randrange, seed
import time

def stress_test(N: int, T: int, M: int, number: str):
    """
    Inverse N matrices aléatoire de taille T x T dont les éléments 
    sont compris entre 1 et M, et retourne le temps d'exécution.
    N : nombre de matrices à inverser
    T : taille des matrices
    M : valeur max des éléments
    """
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
    detailed_scores = []
    detailed_scores.append(stress_test(10, 500, 1000, "1"))
    detailed_scores.append(stress_test(8, 1000, 1000, "2"))
    detailed_scores.append(stress_test(6, 2000, 1000, "3"))
    detailed_scores.append(stress_test(4, 3000, 1000, "4"))
    detailed_scores.append(stress_test(2, 4000, 1000, "5"))
    return detailed_scores

if __name__ == "__main__":
    print("[DEBUT DU PROGRAMME")
    scores = total_score()
    print("[INFO] Scores détaillés :", scores)
    print("[INFO] Score total :", sum(scores))
    print("[FIN DU PROGRAMME]")