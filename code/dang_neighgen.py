
import random
import numpy as np

neighgen_operators = ['cxOnePoint',
                      'cxTwoPoint',
                      'cxUniform',
                      'cxBlend',
                      'cxUniformBlend',
                      'sxSuppress'
                      ]


def cxOnePoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    ind1o = ind1.deepcopy()
    ind2o = ind2.deepcopy()
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2o[cxpoint:], ind1o[cxpoint:]

    return ind1, ind2


def cxTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    ind1o = ind1.deepcopy()
    ind2o = ind2.deepcopy()
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2o[cxpoint1:cxpoint2], ind1o[cxpoint1:cxpoint2]

    return ind1, ind2


def cxUniform(ind1, ind2, indpb):
    size = min(len(ind1), len(ind2))
    ind1o = ind1.deepcopy()
    ind2o = ind2.deepcopy()
    for i in range(size):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2o[i], ind1o[i]

    return ind1, ind2


def cxBlend(ind1, ind2, alpha):
    ind1o = ind1.deepcopy()
    ind2o = ind2.deepcopy()
    for i, (x1, x2) in enumerate(zip(ind1o, ind2o)):
        ind1[i] = (1. - alpha) * x1 + alpha * x2
        ind2[i] = alpha * x1 + (1. - alpha) * x2

    return ind1, ind2


def cxUniformBlend(ind1, ind2, indpb, alpha):
    ind1o = ind1.deepcopy()
    ind2o = ind2.deepcopy()
    for i, (x1, x2) in enumerate(zip(ind1o, ind2o)):
        if random.random() < indpb:
            ind1[i] = (1. - alpha) * x1 + alpha * x2
            ind2[i] = alpha * x1 + (1. - alpha) * x2

    return ind1, ind2


def sxSuppress(ind, base, indpb):
    for i in range(len(ind)):
        if random.random() < indpb:
            idx_base = i if len(base) == len(ind) else 0
            ind[i] = base[idx_base]

    return ind


def dang_neighborhood_generation(x, X_S, n_samples=1000, indpb=0.5, neighgen_op=None, base=None):

    Z = list()
    n_samples_per_support = max(1, n_samples // len(X_S))
    neighgen_op = neighgen_operators if neighgen_op is None else neighgen_op
    neighgen_op_support = [op for op in neighgen_op if op.startswith('cx')]

    for xs in X_S:
        for i in range(n_samples_per_support):
            x_i, xs_i = x.deepcopy(), xs.deepcopy()
            op_id = np.random.choice(neighgen_op_support)

            if op_id == 'cxOnePoint':
                x_i, xs_i = cxOnePoint(x_i, xs_i)
            elif op_id == 'cxTwoPoint':
                x_i, xs_i = cxTwoPoint(x_i, xs_i)
            elif op_id == 'cxUniform':
                x_i, xs_i = cxUniform(x_i, xs_i, indpb)
            elif op_id == 'cxBlend':
                alpha = np.random.choice(np.arange(0.1, 1.0, 0.1))
                x_i, xs_i = cxBlend(x_i, xs_i, alpha)
            elif op_id == 'cxUniformBlend':
                alpha = np.random.choice(np.arange(0.1, 1.0, 0.1))
                x_i, xs_i = cxUniformBlend(x_i, xs_i, indpb, alpha)

            Z.append(x_i)
            Z.append(xs_i)

    if base is not None and 'sxSuppress' in neighgen_op:
        for i in range(n_samples_per_support):
            x_i = x.deepcopy()
            x_i = sxSuppress(x_i, base, indpb)
            Z.append(x_i)

    idx = np.random.choice(len(Z), size=n_samples, replace=False)
    Z = [z for i, z in enumerate(Z) if i in idx]
    return Z

