import matplotlib.pyplot as plt

from dang.config import *
from dang.util import get_dataset
from dang.dang_neighgen import dang_neighborhood_generation

from dang.datatype import *


neighgen_operators = [
    'cxOnePoint',
                      'cxTwoPoint',
                      'cxUniform',
                      # 'cxBlend',
                      # 'cxUniformBlend',
                      'sxSuppress'
                      ]


def main():
    # dataset = 'wdbc'
    # dataset = 'italypower'
    # dataset = 'mnist'
    dataset = '20newsgroups'

    D = get_dataset(dataset, path_dataset, normalize)
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']

    n_classes = D['n_classes']
    data_type = D['data_type']
    print(X_train.shape, data_type)
    if data_type == 'txt':
        X_train, X_test = D['X_train_txt'], D['X_test_txt']

    idx = 0
    if data_type == 'tab':
        X_train_dt = [TabularRecord(x) for x in X_train]
        x_dt = TabularRecord(X_test[idx])

    elif data_type == 'ts':
        print(data_type, D['window_sizes'][0], D['window_steps'][0])
        window_shape = D['window_sizes'][0]
        step = D['window_steps'][0]
        X_train_dt = [TimeSeriesRecord(x, window_shape=window_shape, step=step) for x in X_train]
        x_dt = TimeSeriesRecord(X_test[idx], window_shape=window_shape, step=step)

    elif data_type == 'img':
        print(data_type)
        window_shape = (14, 14) #D['window_sizes'][2]
        step = 7 #D['window_steps'][2][0]
        X_train_dt = [ImageRecord(x, window_shape=window_shape, step=step) for x in X_train[:1000]]
        x_dt = ImageRecord(X_test[idx], window_shape=window_shape, step=step)

    elif data_type == 'txt':
        print(data_type)
        window_shape = 3
        step = 1
        X_train_dt = [TextRecord(x, window_shape=window_shape, step=step, text_length=100) for x in X_train[:1000]]
        x_dt = TextRecord(X_test[idx], window_shape=window_shape, step=step, text_length=100)

    else:
        raise ValueError('Unknown data type %s' % data_type)

    # print(x_dt)
    Z = dang_neighborhood_generation(x_dt, X_train_dt, n_samples=1000, indpb=0.5, neighgen_op=neighgen_operators, base=None)
    print(len(Z))
    # print(Z[3].data.shape)

    # print(x_dt[0].data)
    # plt.imshow(x_dt[0].data, cmap='gray')
    # plt.show()

    # print(Z[0])
    # plt.imshow(Z[1].data, cmap='gray')
    # plt.show()

    # plt.plot(x_dt.data.tolist(), lw=5)
    # for i in range(10):
    #     plt.plot(Z[i*2].data.tolist())
    # plt.show()

    print(' '.join([term for term in x_dt.data if len(term) > 0]))
    print('----')
    for i in range(10):
        print(' '.join([term for term in Z[i].data if len(term) > 0]))
        print('----')








if __name__ == "__main__":
    main()
