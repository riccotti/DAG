import random
import numpy as np

# selezion unfiormamente random e rimpiazza
# funziona su tutti i tipi di dati


def rand_neighborhood_generation(x, X_S, n_samples=1000, indpb=0.5):

    Z = list()
    for i in range(n_samples):
        x_i = x.deepcopy()
        idx = np.random.choice(len(X_S))
        x_b = X_S[idx]
        for j in range(len(x_i)):
            if random.random() < indpb:
                x_i[j] = x_b[j]
        Z.append(x_i)

    return Z

