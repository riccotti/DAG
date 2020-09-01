import random
import numpy as np
from scipy.stats import mode

# calcola media e standard deviation di ogni feature e rimpiazza con valore scelto a caso in quel modo
# funziona su tutti i tipi di dati tranne il testo


def mode_neighborhood_generation(x, X_S, n_samples=1000, indpb=0.5):

    size = len(X_S[0])
    values = list()
    for i in range(size):
        for x_s in X_S:
            values.append(x_s[i])

    values = np.concatenate(values)
    values = values[np.where(values != '')]
    modev, _ = mode(values)

    Z = list()
    for i in range(n_samples):
        x_i = x.deepcopy()
        for j in range(len(x_i)):
            if random.random() < indpb:
                x_i[j] = modev
        Z.append(x_i)

    return Z

