import warnings
warnings.filterwarnings("ignore")

from dang.experiment_evaluate import *
from dang.experiment_tab import run_dang, run_norm, run_rand, run_supp, run_cgan

from scipy.spatial.distance import pdist



def main():

    dataset = 'parkinsons'

    for dataset in ['wdbc', 'diabetes', 'ctg', 'ionoshpere', 'parkinsons', 'sonar', 'vehicle', 'avila']:
    # for dataset in ['gunpoint', 'italypower', 'arrowhead', 'ecg200', 'phalanges', 'electricdevices']:

        method_name = 'norm'

        test_size = 100
        train_size = 10
        n_samples = 1000
        nbr_experiments = 100

        print(datetime.datetime.now(), 'Dataset: %s' % dataset, method_name)
        D = get_dataset(dataset, path_dataset, normalize=None)

        X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
        n_classes = D['n_classes']

        if len(X_test) > test_size:
            idx = np.random.choice(len(X_test), size=test_size, replace=False)
            X_T = X_test[idx]
        else:
            X_T = X_test

        diff_means = list()
        for x in X_T:
            Z_stats, run_time_list = list(), list()
            for i in range(nbr_experiments):
                print(datetime.datetime.now(), '\tSize: %s' % train_size, method_name)

                if len(X_train) > train_size:
                    idx = np.random.choice(len(X_train), size=train_size, replace=False)
                    X_S = X_train[idx]
                else:
                    X_S = X_train

                if method_name == 'dang':
                    Z, run_train = run_dang(X_S, x, n_samples)
                    run_time_list.append(run_train)

                elif method_name == 'rand':
                    Z, run_train = run_rand(X_S, x, n_samples)
                    run_time_list.append(run_train)

                elif method_name == 'supp':
                    Z, run_train = run_supp(X_S, x, n_samples)
                    run_time_list.append(run_train)

                elif method_name == 'norm':
                    Z, run_train = run_norm(X_S, x, n_samples)
                    run_time_list.append(run_train)

                elif method_name == 'cgan':
                    Z, run_train = run_cgan(X_S, x, n_samples)
                    run_time_list.append(run_train)

                else:
                    raise ValueError('Unknown method %s' % method_name)

                Z_stats.append(np.mean(Z, axis=0))

            val = pdist(np.array(Z_stats))
            diff_means.append(np.mean(val))

        eval_stability = {
            'stability_mean': float(np.mean(diff_means)),
            'stability_std': float(np.std(diff_means)),
            'stability_sum': float(np.sum(diff_means)),
            'stability_median': float(np.median(diff_means)),
            'stability_min': float(np.min(diff_means)),
            'stability_max': float(np.max(diff_means)),
        }

        eval_stability['dataset'] = dataset
        eval_stability['method'] = method_name
        eval_stability['n_samples'] = n_samples
        eval_stability['train_size'] = train_size
        eval_stability['test_size'] = test_size

        print(datetime.datetime.now(), 'Storing evaluation')
        store_result(eval_stability, 'tabular_neigh_stability')
        # store_result(eval_stability, 'ts_neigh_stability')


if __name__ == "__main__":
    main()

