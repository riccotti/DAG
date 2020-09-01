

import datetime

from dang.util import *
from dang.config import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():

    for dataset in [
        # 'wdbc', 'diabetes', 'ctg', 'ionoshpere', 'parkinsons', 'sonar', 'vehicle', 'avila',
        # 'gunpoint', 'italypower', 'arrowhead', 'ecg200', 'phalanges', 'electricdevices',
        # 'mnist', 'fashion_mnist', 'cifar10',
        '20newsgroups',
        # 'imdb'
                    ]:

        print(datetime.datetime.now(), 'Dataset: %s' % dataset)

        if dataset == '20newsgroups':
            categories = ['alt.atheism', 'talk.religion.misc']
        else:
            categories = None

        D = get_dataset(dataset, path_dataset, normalize=None, categories=categories)

        X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']

        dataset_type = datasets[dataset][1]

        if dataset_type == 'img':
            s0, s1, s2 = X_train.shape
            X_train = X_train.reshape(s0, s1 * s2)

            s0, s1, s2 = X_test.shape
            X_test = X_test.reshape(s0, s1 * s2)

        clf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=16,
                                     min_samples_split=10, min_samples_leaf=5, n_jobs=8)
        clf.fit(X_train, y_train)
        pickle.dump(clf, open(path_clf + '%s_%s.pickle' % (dataset, 'RF'), 'wb'))
        y_pred = clf.predict(X_test)
        print('Accuracy', accuracy_score(y_test, y_pred))
        print('')


if __name__ == "__main__":
    main()
