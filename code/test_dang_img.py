import matplotlib.pyplot as plt

from dang.config import *
from dang.util import get_dataset
from dang.dang_neighgen import dang_neighborhood_generation
from dang.rand_neighgen import rand_neighborhood_generation
from dang.supp_neighgen import supp_neighborhood_generation
from dang.norm_neighgen import norm_neighborhood_generation

from dang.datatype import *


neighgen_operators = [
    'cxOnePoint',
    'cxTwoPoint',
    'cxUniform',
    'cxBlend',
    'cxUniformBlend',
    'sxSuppress']


def main():

    dataset = 'mnist'
    train_size = 1000

    D = get_dataset(dataset, path_dataset)
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']

    data_type = D['data_type']
    if data_type == 'txt':
        X_train, X_test = D['X_train_txt'], D['X_test_txt']

    if len(X_train) > train_size:
        idx = np.random.choice(len(X_train), size=train_size, replace=False)
        X_train = X_train[idx]

    idx = 0
    if data_type == 'tab':
        X_train_dt = [TabularRecord(x) for x in X_train]
        x_dt = TabularRecord(X_test[idx])

    elif data_type == 'ts':
        window_shape = D['window_sizes'][0]
        step = D['window_steps'][0]
        X_train_dt = [TimeSeriesRecord(x, window_shape=window_shape, step=step) for x in X_train]
        x_dt = TimeSeriesRecord(X_test[idx], window_shape=window_shape, step=step)

    elif data_type == 'img':
        # X_train = X_train * 255.0
        # X_train = (X_train - 127.5) / 127.5
        window_shape = D['window_sizes'][2]
        step = D['window_steps'][2][0]
        X_train_dt = [ImageRecord(x, window_shape=window_shape, step=step) for x in X_train]
        x_dt = ImageRecord(X_test[idx], window_shape=window_shape, step=step)

    elif data_type == 'txt':
        window_shape = 3
        step = 1
        X_train_dt = [TextRecord(x, window_shape=window_shape, step=step, text_length=100) for x in X_train]
        x_dt = TextRecord(X_test[idx], window_shape=window_shape, step=step, text_length=100)

    else:
        raise ValueError('Unknown data type %s' % data_type)

    Z = dang_neighborhood_generation(x_dt, X_train_dt, n_samples=1000, indpb=0.5, neighgen_op=neighgen_operators, base=None)

    # Z = rand_neighborhood_generation(x_dt, X_train_dt, n_samples=1000, indpb=0.5)

    # Z = supp_neighborhood_generation(x_dt, 0.8, n_samples=1000, indpb=0.5)

    # Z = norm_neighborhood_generation(x_dt, X_train_dt, n_samples=1000, indpb=0.5)

    print(len(Z), np.mean(X_train), np.mean(Z))

    plt.imshow(x_dt[0].data, cmap='gray')
    plt.imshow(Z[1].data, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
