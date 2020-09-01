import time
import json
import pickle
import datetime
import numpy as np
import pandas as pd

from ctgan import CTGANSynthesizer

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from dang.config import *
from dang.util import get_dataset


clf_list = {
    'RF': RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=0.02,
                                 min_samples_leaf=0.01, max_features='auto', n_jobs=-1),
    'AB': AdaBoostClassifier(n_estimators=100),
    'NN': MLPClassifier(hidden_layer_sizes=(32, 64, 128), activation='relu', solver='adam', alpha=0.0001,
                        batch_size='auto', learning_rate='adaptive', max_iter=200, shuffle=True, tol=1e-4,
                        early_stopping=True),
    'SVM': SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', tol=1e-3)
}

# ctgan = CTGANSynthesizer()
#
# dataset = 'diabetes'
#
# D = get_dataset(dataset, path_dataset, normalize=None)
#
# ctgan.fit(D['X_train'], epochs=10)
#
# X_real = D['X_train']
# X_fake = ctgan.sample(len(D['X_train']))
#
# print(X_fake)
#
# clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=0.02,
#                                  min_samples_leaf=0.01, max_features='auto', n_jobs=-1)
#
# y_real = [1] * len(X_real)
# y_fake = [0] * len(X_fake)
#
# X_rf = np.concatenate([X_real, X_fake])
# y_rf = np.concatenate([y_real, y_fake])
#
# X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, train_size=0.7, stratify=y_rf)
#
# clf.fit(X_train, y_train)
# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# acc_train = accuracy_score(y_train, y_pred_train)
# acc_test = accuracy_score(y_test, y_pred_test)
#
# print(acc_train)
# print(acc_test)


def store_result(eval_obj, filename):
    fout = open(filename, 'a')
    json_str = ('%s\n' % json.dumps(eval_obj))
    fout.write(json_str)
    fout.close()


def main():

    dataset = 'diabetes'
    epochs = 300
    train_size = 0.7

    print(datetime.datetime.now(), 'Dataset: %s' % dataset)
    D = get_dataset(dataset, path_dataset, normalize=None)

    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    # n_classes = D['n_classes']
    n_features = D['n_features']
    feature_names = D['feature_names']
    class_name = D['class_name']

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    Xy_train = np.hstack((X_train, y_train.reshape(-1, 1)))

    print(datetime.datetime.now(), 'Training CTGAN')
    ctgan = CTGANSynthesizer(embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256), l2scale=1e-6, batch_size=500)
    ts = time.time()
    ctgan.fit(Xy_train, epochs=epochs, discrete_columns=[n_features+1])
    cgan_fit_time = time.time() - ts

    n_fake_instances = len(Xy_train)

    print(datetime.datetime.now(), 'Generating synthetic data')
    ts = time.time()
    Xy_fake = ctgan.sample(n_fake_instances)
    cgan_gen_time = time.time() - ts

    # print('F 0', np.mean(Xy_fake[:, 0]), np.min(Xy_fake[:,0]), np.max(Xy_fake[:,0]))
    # print('F 1', np.mean(Xy_fake[:, 1]), np.min(Xy_fake[:, 1]), np.max(Xy_fake[:, 1]))
    #
    # print('R 0', np.mean(X_train[:, 0]), np.min(X_train[:, 0]), np.max(X_train[:, 0]))
    # print('R 1', np.mean(X_train[:, 1]), np.min(X_train[:, 1]), np.max(X_train[:, 1]))
    # return -1

    print(datetime.datetime.now(), 'Storing synthetic data')
    df = pd.DataFrame(data=Xy_fake, columns=feature_names + [class_name])
    df.to_csv(path_syht_dataset + '%s.csv' % dataset, index=False)

    X_fake = Xy_fake[:, :-1]
    X_real = X_train

    y_real = np.ones(len(X_real))
    y_fake = np.zeros(len(X_fake))

    X_rf = np.concatenate([X_real, X_fake])
    y_rf = np.concatenate([y_real, y_fake])

    X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, train_size=train_size, stratify=y_rf)

    res_dict = dict()

    for clf_name, clf in clf_list.items():
        print(datetime.datetime.now(), 'Training %s' % clf_name)
        ts = time.time()
        clf.fit(X_rf_train, y_rf_train)
        disc_fit_time = time.time() - ts
        pickle.dump(clf, open(path_discr + '%s_%s.pickle' % (dataset, clf_name), 'wb'))

        y_pred_train = clf.predict(X_rf_train)
        y_pred_test = clf.predict(X_rf_test)
        acc_train = accuracy_score(y_rf_train, y_pred_train)
        acc_test = accuracy_score(y_rf_test, y_pred_test)
        res_dict['%s_acc_train' % clf_name] = acc_train
        res_dict['%s_acc_test' % clf_name] = acc_test
        res_dict['%s_disc_fit_time' % clf_name] = disc_fit_time
        print(datetime.datetime.now(), '\taccuracy %.3f, %.3f' % (acc_train, acc_test))

    res_dict['dataset'] = dataset
    res_dict['cgan_fit_time'] = cgan_fit_time
    res_dict['cgan_gen_time'] = cgan_gen_time

    print(datetime.datetime.now(), 'Storing evaluation')
    store_result(res_dict, path_ctgan_eval + 'tabular.json')


if __name__ == "__main__":
    main()
