import datetime
import matplotlib.pyplot as plt

from dang.config import *
from dang.util import get_dataset
from dang.dang_neighgen import dang_neighborhood_generation

from dang.datatype import *


neighgen_operators = [
    'cxOnePoint',
    'cxTwoPoint',
    'cxUniform',
    'cxBlend',
    'cxUniformBlend',
    'sxSuppress']


def main():
    dataset = 'wdbc'
    train_size = 1000

    D = get_dataset(dataset, path_dataset, normalize)
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']

    data_type = D['data_type']

    if data_type == 'txt':
        X_train, X_test = D['X_train_txt'], D['X_test_txt']

    if len(X_train) > train_size:
        idx = np.random.choice(len(X_train), size=train_size, replace=False)
        X_train = X_train[idx]
    print(dataset, X_train.shape, data_type)

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

    print(x_dt.data)
    print('----')
    for i in range(10):
        print(Z[i])
        print('----')


def main():
    method = 'dang'

    for dataset in tab_datasets:
        print(datetime.datetime.now(), 'Dataset: %s' % dataset)

        D = get_dataset(dataset, path_dataset, normalize)

        run_experiment(D, dataset, method)
        print('')
        # break


if __name__ == "__main__":
    main()

