import random
import numpy as np

# selezion unfiormamente random e rimpiazza
# funziona su tutti i tipi di dati


def supp_neighborhood_generation(x, base_value, n_samples=1000, indpb=0.5):

    Z = list()
    for i in range(n_samples):
        x_i = x.deepcopy()
        for j in range(len(x_i)):
            if random.random() < indpb:
                x_i[j] = base_value
        Z.append(x_i)

    return Z

