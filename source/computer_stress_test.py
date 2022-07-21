import numpy as np
from random import randrange, seed
import time

# CONSTANTES
N = 10 # nombre de matrices à inverser
T = 1000 # taille des matrices
M = 100 # valeur max des éléments

start = time.time()
print("[Début du programme]")

s = 0 # seed fixée pour avoir toujours les mêmes résultats
for i in range(N):
    s += 12345
    seed(s)

    # Création d'une matrice de taille TxT
    A = np.array([[randrange(1, M+1) for j in range(T)] for i in range(T)])
    B = np.linalg.inv(A)
    print("     ["+str(i+1)+"/"+str(N)+"]")

print("[TEMPS TOTAL]", round(time.time() - start, 2), "secondes")
print("[Fin du programme]")