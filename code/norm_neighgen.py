import random
import numpy as np

# calcola media e standard deviation di ogni feature e rimpiazza con valore scelto a caso in quel modo
# funziona su tutti i tipi di dati tranne il testo


def norm_neighborhood_generation(x, X_S, n_samples=1000, indpb=0.5):

    size = len(X_S[0])
    means = list()
    stds = list()
    for i in range(size):
        values = list()
        for x_s in X_S:
            values.append(x_s[i])
        means.append(np.mean(values, axis=0))
        stds.append(np.std(values, axis=0))

    Z = list()
    for i in range(n_samples):
        x_i = x.deepcopy()
        for j in range(len(x_i)):
            if random.random() < indpb:
                x_i[j] = np.random.normal(loc=means[j], scale=stds[j])
        Z.append(x_i)

    return Z

