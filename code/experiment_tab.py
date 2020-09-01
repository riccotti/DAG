import warnings
warnings.filterwarnings("ignore")

import time

from ctgan import CTGANSynthesizer

from dang.datatype import *
from dang.experiment_evaluate import *
from dang.dang_neighgen import dang_neighborhood_generation
from dang.rand_neighgen import rand_neighborhood_generation
from dang.supp_neighgen import supp_neighborhood_generation
from dang.norm_neighgen import norm_neighborhood_generation


def run_dang(X_S, x_T, n_samples):
    neighgen_operators = ['cxOnePoint', 'cxTwoPoint', 'cxUniform', 'cxBlend', 'cxUniformBlend', 'sxSuppress']
    X_S_dt = [TabularRecord(x) for x in X_S]

    x = x_T
    ts = time.time()
    x_dt = TabularRecord(x)
    Z_dt = dang_neighborhood_generation(x_dt, X_S_dt, n_samples=n_samples, indpb=0.5,
                                        neighgen_op=neighgen_operators, base=None)
    Z = [z.data for z in Z_dt]
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_rand(X_S, x_T, n_samples):
    X_S_dt = [TabularRecord(x) for x in X_S]

    x = x_T
    ts = time.time()
    x_dt = TabularRecord(x)
    Z_dt = rand_neighborhood_generation(x_dt, X_S_dt, n_samples=n_samples, indpb=0.5)
    Z = [z.data for z in Z_dt]
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_supp(X_S, x_T, n_samples):
    base_value = 0.0

    x = x_T
    ts = time.time()
    x_dt = TabularRecord(x)
    # base_dt = TabularRecord(base)
    Z_dt = supp_neighborhood_generation(x_dt, base_value, n_samples=n_samples, indpb=0.5)
    Z = [z.data for z in Z_dt]
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_norm(X_S, x_T, n_samples):
    X_S_dt = [TabularRecord(x) for x in X_S]

    x = x_T
    ts = time.time()
    x_dt = TabularRecord(x)
    Z_dt = norm_neighborhood_generation(x_dt, X_S_dt, n_samples=n_samples, indpb=0.5)
    Z = [z.data for z in Z_dt]
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_cgan(X_S, x_T, n_samples):
    X_train = np.vstack([X_S, x_T])
    ctgan = CTGANSynthesizer(embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256), l2scale=1e-6, batch_size=500)
    ts = time.time()
    ctgan.fit(X_train, epochs=300)
    Z = ctgan.sample(n_samples)
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_experiment(D, dataset_name, method_name, n_samples, train_size, test_size):
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    n_classes = D['n_classes']

    if len(X_test) > test_size:
        idx = np.random.choice(len(X_test), size=test_size, replace=False)
        X_T = X_test[idx]
    else:
        X_T = X_test

    Z_list, run_time_list = list(), list()

    for x in X_T:
        if len(X_train) > train_size:
            idx = np.random.choice(len(X_train), size=train_size, replace=False)
            X_S = X_train[idx]
        else:
            X_S = X_train

        if method_name == 'dang':
            Z, run_train = run_dang(X_S, x, n_samples)
            Z_list.append(Z)
            run_time_list.append(run_train)

        elif method_name == 'rand':
            Z, run_train = run_rand(X_S, x, n_samples)
            Z_list.append(Z)
            run_time_list.append(run_train)

        elif method_name == 'supp':
            Z, run_train = run_supp(X_S, x, n_samples)
            Z_list.append(Z)
            run_time_list.append(run_train)

        elif method_name == 'norm':
            Z, run_train = run_norm(X_S, x, n_samples)
            Z_list.append(Z)
            run_time_list.append(run_train)

        elif method_name == 'cgan':
            Z, run_train = run_cgan(X_S, x, n_samples)
            Z_list.append(Z)
            run_time_list.append(run_train)

        else:
            raise ValueError('Unknown method %s' % method_name)

    eval_ng = evaluate(X_train, Z_list, n_classes, dataset_name,
                       D=None, max_nbr_dimensions=None, max_nbr_instances_lof=None, verbose=False)

    eval_ng['dataset'] = dataset_name
    eval_ng['method'] = method_name
    eval_ng['n_samples'] = n_samples
    eval_ng['train_size'] = train_size
    eval_ng['test_size'] = test_size
    eval_ng['time_mean'] = np.mean(run_time_list)
    eval_ng['time_std'] = np.mean(run_time_list)

    print(datetime.datetime.now(), 'Storing evaluation')
    store_result(eval_ng, 'tabular_neigh')
    return 0


def main():

    dataset = 'parkinsons'

    for dataset in ['wdbc', 'diabetes', 'ctg', 'ionoshpere', 'parkinsons', 'sonar', 'vehicle', 'avila']:

        method = 'cgan'
        test_size = 100                                                         # nbr instances to explain
        train_sizes = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]      # nbr instances in the support set
        n_samples_list = [10, 100, 1000, 10000]                                 # nbr instances generated

        # test_size = 100
        # train_sizes = [10, 25, 50]
        # n_samples_list = [1000]

        print(datetime.datetime.now(), 'Dataset: %s' % dataset, method)
        D = get_dataset(dataset, path_dataset, normalize=None)

        for train_size in train_sizes:
            print(datetime.datetime.now(), '\tSize: %s' % train_size, method)

            for n_samples in n_samples_list:
                print(datetime.datetime.now(), '\t\tN Samples: %s' % n_samples, method)
                run_experiment(D, dataset, method, n_samples, train_size, test_size)
                print('')

            if len(D['X_train']) < train_size:
                break


if __name__ == "__main__":
    main()

