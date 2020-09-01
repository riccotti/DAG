import warnings
warnings.filterwarnings("ignore")

import time
from sklearn.preprocessing import LabelEncoder

from dang.txt_gan import GAN, clean_texts, texts2texts_nextword

from dang.datatype import *
from dang.experiment_evaluate import *
from dang.dang_neighgen import dang_neighborhood_generation
from dang.rand_neighgen import rand_neighborhood_generation
from dang.supp_neighgen import supp_neighborhood_generation
from dang.mode_neighgen import mode_neighborhood_generation


def run_dang(X_S, x_T, n_samples, window_shape, step, text_length):
    neighgen_operators = ['cxOnePoint', 'cxTwoPoint', 'cxUniform', 'sxSuppress']
    X_S_dt = [TextRecord(x, window_shape=window_shape, step=step, text_length=text_length) for x in X_S]

    x = x_T
    ts = time.time()
    x_dt = TextRecord(x, window_shape=window_shape, step=step, text_length=text_length)
    Z_dt = dang_neighborhood_generation(x_dt, X_S_dt, n_samples=n_samples, indpb=0.5,
                                        neighgen_op=neighgen_operators, base=None)
    Z = [z.data for z in Z_dt]
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_rand(X_S, x_T, n_samples, window_shape, step, text_length):
    X_S_dt = [TextRecord(x, window_shape=window_shape, step=step, text_length=text_length) for x in X_S]

    x = x_T
    ts = time.time()
    x_dt = TextRecord(x, window_shape=window_shape, step=step, text_length=text_length)
    Z_dt = rand_neighborhood_generation(x_dt, X_S_dt, n_samples=n_samples, indpb=0.5)
    Z = [z.data for z in Z_dt]
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_supp(X_S, x_T, n_samples, window_shape, step, text_length):
    base_value = ''

    x = x_T
    ts = time.time()
    x_dt = TextRecord(x, window_shape=window_shape, step=step, text_length=text_length)
    Z_dt = supp_neighborhood_generation(x_dt, base_value, n_samples=n_samples, indpb=0.5)
    Z = [z.data for z in Z_dt]
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_norm(X_S, x_T, n_samples, window_shape, step, text_length):
    X_S_dt = [TextRecord(x, window_shape=window_shape, step=step, text_length=text_length) for x in X_S]

    x = x_T
    ts = time.time()
    x_dt = TextRecord(x, window_shape=window_shape, step=step, text_length=text_length)
    Z_dt = mode_neighborhood_generation(x_dt, X_S_dt, n_samples=n_samples, indpb=0.5)
    Z = [z.data for z in Z_dt]
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_cgan(X_S, x_T, n_samples, D):
    x_T = np.array([x_T])
    X_train = np.concatenate([X_S, x_T])
    maxlen = 3
    step = 1

    X_train = clean_texts(X_train, min_chars=10)
    text_lengths = [len(x) for x in X_train]

    X, y, words, words_indices, indices_words = texts2texts_nextword(X_train, maxlen=maxlen, step=step)
    nbr_terms = len(words)
    start_texts_idx = np.random.choice(len(X_train), min(10, len(X_train)), replace=False)
    start_texts = [X_train[i] for i in start_texts_idx]

    gan = GAN(nbr_terms, maxlen, latent_dim=100, term_indices=words_indices, indices_term=indices_words,
              txt_path=None, verbose=False,
              start_texts=start_texts, text_lengths=text_lengths)

    ts = time.time()
    gan.fit(X, y, epochs=10, batch_size=128)
    Z = gan.sample(n_samples, t=10, start_texts=start_texts,
                   diversity_list=[0.2, 0.5, 0.7, 1.0, 1.2])
    # print(Z[0].tolist())
    # return None
    run_train = time.time() - ts

    return Z, run_train


def run_experiment(D, dataset_name, method_name, n_samples, train_size, test_size):
    X_train, y_train, X_test, y_test = D['X_train_txt'], D['y_train'], D['X_test_txt'], D['y_test']
    n_classes = D['n_classes']
    window_shape = 3
    step = 3
    text_length = 1000

    if len(X_test) > test_size:
        idx = np.random.choice(len(X_test), size=test_size, replace=False)
        X_T = np.array(X_test)[idx]
    else:
        X_T = X_test

    Z_list, run_time_list = list(), list()

    # print(datetime.datetime.now(), 'Started Generation')
    for i, x in enumerate(X_T):
        # print(datetime.datetime.now(), 'Generation %d/%d' % (i+1, len(X_T)))
        if len(X_train) > train_size:
            idx = np.random.choice(len(X_train), size=train_size, replace=False)
            X_S = np.array(X_train)[idx]
        else:
            X_S = X_train

        if method_name == 'dang':
            Z, run_train = run_dang(X_S, x, n_samples, window_shape, step, text_length)
            Z = [' '.join([term for term in z if len(term) > 0]) for z in Z]
            Z_list.append(Z)
            run_time_list.append(run_train)

        elif method_name == 'rand':
            Z, run_train = run_rand(X_S, x, n_samples, window_shape, step, text_length)
            Z = [' '.join([term for term in z if len(term) > 0]) for z in Z]
            Z_list.append(Z)
            run_time_list.append(run_train)

        elif method_name == 'supp':
            Z, run_train = run_supp(X_S, x, n_samples, window_shape, step, text_length)
            Z = [' '.join([term for term in z if len(term) > 0]) for z in Z]
            Z_list.append(Z)
            run_time_list.append(run_train)

        elif method_name == 'norm':
            Z, run_train = run_norm(X_S, x, n_samples, window_shape, step, text_length)
            Z = [' '.join([term for term in z if len(term) > 0]) for z in Z]
            Z_list.append(Z)
            run_time_list.append(run_train)

        elif method_name == 'cgan':
            Z, run_train = run_cgan(X_S, x, n_samples, D)
            Z_list.append(Z)
            run_time_list.append(run_train)

        else:
            raise ValueError('Unknown method %s' % method_name)

    eval_ng = evaluate(X_train, Z_list, n_classes, dataset_name, D,
                       max_nbr_dimensions=100, max_nbr_instances_lof=2500, verbose=True)

    eval_ng['dataset'] = dataset_name
    eval_ng['method'] = method_name
    eval_ng['n_samples'] = n_samples
    eval_ng['train_size'] = train_size
    eval_ng['test_size'] = test_size
    eval_ng['time_mean'] = np.mean(run_time_list)
    eval_ng['time_std'] = np.mean(run_time_list)

    print(datetime.datetime.now(), 'Storing evaluation')
    store_result(eval_ng, 'text_neigh')
    return 0


def main():

    dataset = '20newsgroups'

    for dataset in [
        # '20newsgroups',
                    'imdb']:
        method = 'cgan'
        test_size = 100                                                         # nbr instances to explain
        train_sizes = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]      # nbr instances in the support set
        # n_samples_list = [10, 100, 1000, 10000]                                 # nbr instances generated
        #
        # test_size = 100
        # train_sizes = [10, 25, 50]
        n_samples_list = [1000]

        print(datetime.datetime.now(), 'Dataset: %s' % dataset)

        if dataset == '20newsgroups':
            categories = ['alt.atheism', 'talk.religion.misc']
        else:
            categories = None

        D = get_dataset(dataset, path_dataset, categories=categories)

        for train_size in train_sizes:
            print(datetime.datetime.now(), '\tSize: %s' % train_size)

            for n_samples in n_samples_list:
                print(datetime.datetime.now(), '\t\tN Samples: %s' % n_samples)
                run_experiment(D, dataset, method, n_samples, train_size, test_size)
                print('')

            if len(D['X_train']) < train_size:
                break


if __name__ == "__main__":
    main()

