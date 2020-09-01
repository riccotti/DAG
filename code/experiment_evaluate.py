import json
import datetime
from dang.util import *
from dang.config import *

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.decomposition import PCA

from collections import defaultdict


def store_result(eval_obj, method_name):
    fout = open(path_eval + '%s.json' % method_name, 'a')
    for k, v in eval_obj.items():
        # print(k, type(v))
        if not isinstance(v, str):
            eval_obj[k] = float(v)
    json_str = ('%s\n' % json.dumps(eval_obj))
    fout.write(json_str)
    fout.close()


clf_list = {
    'RF': RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=0.02,
                                 min_samples_leaf=0.01, max_features='auto', n_jobs=-1),
    'AB': AdaBoostClassifier(n_estimators=100),
    'NN': MLPClassifier(hidden_layer_sizes=(32, 64, 128), activation='relu', solver='adam', alpha=0.0001,
                        batch_size='auto', learning_rate='adaptive', max_iter=200, shuffle=True, tol=1e-4,
                        early_stopping=True),
    'SVM': SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', tol=1e-3)
}


def evaluate(X, Z_list, n_classes, dataset, D=None, max_nbr_dimensions=100, max_nbr_instances_lof=1000, verbose=False):
    n_clusters = 5 * n_classes

    if D is not None:
        vectorizer = D['vectorizer']
        X = vectorizer.transform(X).toarray()

    is_image = False
    if X.ndim > 2:
        is_image = True
        s0, s1, s2 = X.shape
        X_r = X.reshape(s0, s1 * s2)
    else:
        X_r = X

    if verbose:
        print(datetime.datetime.now(), 'Scaling')
    scaler = StandardScaler()
    X_r_s = scaler.fit_transform(X_r)

    pca = None
    if max_nbr_dimensions is not None and X_r_s.shape[1] > max_nbr_dimensions:
        if verbose:
            print(datetime.datetime.now(), 'PCA')
        pca = PCA(n_components=max_nbr_dimensions)
        pca.fit(X_r_s)
        X_r_s = pca.transform(X_r_s)

    if verbose:
        print(datetime.datetime.now(), 'K-Means')
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10,
                    max_iter=250, tol=1e-4, n_jobs=-1)
    kmeans.fit(X_r_s)

    if max_nbr_instances_lof is not None and len(X_r_s) > max_nbr_instances_lof:
        idx = np.random.choice(len(X_r_s), size=max_nbr_instances_lof, replace=False)
        X_r_s_lof = X_r_s[idx]
    else:
        X_r_s_lof = X_r_s

    if verbose:
        print(datetime.datetime.now(), 'LOF')

    lof = LocalOutlierFactor(n_neighbors=5, novelty=True, n_jobs=8)
    lof.fit(X_r_s_lof)

    # if len(X) > n_samples:
    #     idx = np.random.choice(len(X), size=n_samples, replace=False)
    #     X_real = X[idx]
    # else:
    #     X_real = X
    # y_real = [1] * len(X_real)

    distances_list = list()
    silhouette_list = list()
    lof_list = list()
    nbr_clusters_list = list()
    cluster_purity_list = list()
    nbr_classes_list = list()
    class_purity_list = list()
    accuracy_dict = defaultdict(list)
    deltas_list = defaultdict(list)

    real_mean = np.mean(X_r_s)
    real_std = np.std(X_r_s)
    real_min = np.min(X_r_s)
    real_max = np.max(X_r_s)
    real_median = np.median(X_r_s)

    # rus = RandomUnderSampler()

    for i, Z in enumerate(Z_list):
        # print(datetime.datetime.now(), i)

        if D is not None:
            vectorizer = D['vectorizer']
            Z = vectorizer.transform(Z).toarray()

        # if isinstance(Z, np.ndarray) and Z.ndim > 2:
        if is_image:
            if isinstance(Z, list):
                Z = np.array(Z)
            s0, s1, s2 = Z.shape
            Z_r = Z.reshape(s0, s1 * s2)
        else:
            Z_r = Z

        Z_s = scaler.transform(Z_r)
        Z_s_o = Z_s
        if pca is not None:
            Z_s = pca.transform(Z_s)

        labels = kmeans.predict(Z_s)
        dist = kmeans.transform(Z_s)

        # print('qui')
        if verbose:
            print(datetime.datetime.now(), 'distances', i)
        distances = [d[l] for l, d in zip(labels, dist)]
        distances_list.append([np.mean(distances), np.std(distances), np.sum(distances),
                               np.median(distances), np.min(distances), np.max(distances)])

        # print('quo')
        if verbose:
            print(datetime.datetime.now(), 'silhouette', i)
        if 1 < len(np.unique(labels)) < len(labels):
            sil_values = silhouette_samples(Z_s, labels)
            silhouette_list.append([np.mean(sil_values), np.std(sil_values), np.sum(sil_values),
                                   np.median(sil_values), np.min(sil_values), np.max(sil_values)])
        else:
            silhouette_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # print('qua')
        if verbose:
            print(datetime.datetime.now(), 'lof', i)
        lof_values = -lof.score_samples(Z_s)
        lof_list.append([np.mean(lof_values), np.std(lof_values), np.sum(lof_values),
                         np.median(lof_values), np.min(distances), np.max(lof_values)])

        nbr_clusters_list.append(len(np.unique(labels)))

        _, counts = np.unique(labels, return_counts=True)
        cluster_purity = np.max(counts) / np.sum(counts)
        cluster_purity_list.append(cluster_purity)

        if verbose:
            print(datetime.datetime.now(), 'classes', i)
        clf = pickle.load(open(path_clf + '%s_%s.pickle' % (dataset, 'RF'), 'rb'))
        y_pred = clf.predict(Z_s_o)

        nbr_classes_list.append(len(np.unique(y_pred)))

        _, counts = np.unique(y_pred, return_counts=True)
        class_purity = np.max(counts) / np.sum(counts)
        class_purity_list.append(class_purity)

        # X_fake = Z_s
        # y_fake = [0] * len(X_fake)
        #
        # X_rf = np.concatenate([X_real, X_fake])
        # y_rf = np.concatenate([y_real, y_fake])
        #
        # X_rf, y_rf = rus.fit_resample(X_rf, y_rf)
        #
        # X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, train_size=0.7, stratify=y_rf)
        #
        # for clf_name, clf in clf_list.items():
        #     # print(clf_name)
        #     clf.fit(X_train, y_train)
        #     y_pred_train = clf.predict(X_train)
        #     y_pred_test = clf.predict(X_test)
        #     acc_train = accuracy_score(y_train, y_pred_train)
        #     acc_test = accuracy_score(y_test, y_pred_test)
        #     accuracy_dict['%s_acc_train' % clf_name].append(acc_train)
        #     accuracy_dict['%s_acc_test' % clf_name].append(acc_test)

        fake_mean = np.mean(Z_s)
        fake_std = np.std(Z_s)
        fake_min = np.min(Z_s)
        fake_max = np.max(Z_s)
        fake_median = np.median(Z_s)

        if verbose:
            print(datetime.datetime.now(), 'deltas', i)
        deltas_list['delta_mean'] = np.abs(fake_mean - real_mean)
        deltas_list['delta_std'] = np.abs(fake_std - real_std)
        deltas_list['delta_min'] = np.abs(fake_min - real_min)
        deltas_list['delta_max'] = np.abs(fake_max - real_max)
        deltas_list['delta_median'] = np.abs(fake_median - real_median)

        y_true = np.zeros(len(Z_s))

        if verbose:
            print(datetime.datetime.now(), 'discriminator', i)

        for clf_name in clf_list:
            clf = pickle.load(open(path_discr + '%s_%s.pickle' % (dataset, clf_name), 'rb'))
            data_type = datasets[dataset]
            if data_type == 'img':
                Z_s_o = Z_s_o * 255.0
                Z_s_o = (Z_s_o - 127.5) / 127.5
            y_pred = clf.predict(Z_s_o)
            acc = accuracy_score(y_true, y_pred)
            accuracy_dict['%s_accuracy' % clf_name].append(acc)

    distances_list = np.array(distances_list)
    silhouette_list = np.array(silhouette_list)
    lof_list = np.array(lof_list)

    eval_ng = {
        'dist_mean': np.mean(distances_list[:, 0]),
        'dist_std': np.mean(distances_list[:, 1]),
        'dist_sum': np.mean(distances_list[:, 2]),
        'dist_median': np.mean(distances_list[:, 3]),
        'dist_min': np.mean(distances_list[:, 4]),
        'dist_max': np.mean(distances_list[:, 5]),

        'sil_mean': np.mean(silhouette_list[:, 0]),
        'sil_std': np.mean(silhouette_list[:, 1]),
        'sil_sum': np.mean(silhouette_list[:, 2]),
        'sil_median': np.mean(silhouette_list[:, 3]),
        'sil_min': np.mean(silhouette_list[:, 4]),
        'sil_max': np.mean(silhouette_list[:, 5]),

        'lof_mean': np.mean(lof_list[:, 0]),
        'lof_std': np.mean(lof_list[:, 1]),
        'lof_sum': np.mean(lof_list[:, 2]),
        'lof_median': np.mean(lof_list[:, 3]),
        'lof_min': np.mean(lof_list[:, 4]),
        'lof_max': np.mean(lof_list[:, 5]),

        'cluster_purity_mean': float(np.mean(cluster_purity_list)),
        'cluster_purity_std': float(np.std(cluster_purity_list)),
        'cluster_purity_sum': float(np.sum(cluster_purity_list)),
        'cluster_purity_median': float(np.median(cluster_purity_list)),
        'cluster_purity_min': float(np.min(cluster_purity_list)),
        'cluster_purity_max': float(np.max(cluster_purity_list)),

        'nbr_clus_mean': float(np.mean(nbr_clusters_list)),
        'nbr_clus_std': float(np.std(nbr_clusters_list)),
        'nbr_clus_sum': float(np.sum(nbr_clusters_list)),
        'nbr_clus_median': float(np.median(nbr_clusters_list)),
        'nbr_clus_min': float(np.min(nbr_clusters_list)),
        'nbr_clus_max': float(np.max(nbr_clusters_list)),
        'nbr_clusters': n_clusters,

        'class_purity_mean': float(np.mean(class_purity_list)),
        'class_purity_std': float(np.std(class_purity_list)),
        'class_purity_sum': float(np.sum(class_purity_list)),
        'class_purity_median': float(np.median(class_purity_list)),
        'class_purity_min': float(np.min(class_purity_list)),
        'class_purity_max': float(np.max(class_purity_list)),

        'nbr_class_mean': float(np.mean(nbr_classes_list)),
        'nbr_class_std': float(np.std(nbr_classes_list)),
        'nbr_class_sum': float(np.sum(nbr_classes_list)),
        'nbr_class_median': float(np.median(nbr_classes_list)),
        'nbr_class_min': float(np.min(nbr_classes_list)),
        'nbr_class_max': float(np.max(nbr_classes_list)),
        'nbr_classs': n_classes,
    }

    for acc_name in accuracy_dict:
        eval_ng[acc_name] = np.mean(accuracy_dict[acc_name])

    for delta_name in deltas_list:
        eval_ng[delta_name] = np.mean(deltas_list[delta_name])

    return eval_ng
