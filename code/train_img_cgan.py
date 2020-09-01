import time
import json
import pickle
import datetime
import numpy as np
import pandas as pd

from dang.img_cgan import CGAN

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dang.config import *
from dang.util import get_dataset

from keras import backend as K


clf_list = {
    'RF': RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=0.02,
                                 min_samples_leaf=0.01, max_features='auto', n_jobs=-1),
    'AB': AdaBoostClassifier(n_estimators=100),
    'NN': MLPClassifier(hidden_layer_sizes=(32, 64, 128), activation='relu', solver='adam', alpha=0.0001,
                        batch_size='auto', learning_rate='adaptive', max_iter=200, shuffle=True, tol=1e-4,
                        early_stopping=True),
    'SVM': SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', tol=1e-3)
}


def store_result(eval_obj, filename):
    fout = open(filename, 'a')
    json_str = ('%s\n' % json.dumps(eval_obj))
    fout.write(json_str)
    fout.close()


def main():

    print(K.tensorflow_backend._get_available_gpus())

    dataset = 'cifar10'
    epochs = 20000
    train_size = 0.7
    categories = None
    latent_dim = 100

    print(datetime.datetime.now(), 'Dataset: %s' % dataset)
    D = get_dataset(dataset, path_dataset, categories=categories)

    X_train, y_train, _, _ = D['X_train'], D['y_train'], D['X_test'], D['y_test']

    X_train = X_train * 255.0
    X_train = (X_train - 127.5) / 127.5

    n_classes = D['n_classes']
    # n_features = D['n_features']
    # feature_names = D['feature_names']
    # class_name = D['class_name']
    img_rows, img_cols = D['w'], D['h']
    channels = 1

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    # y_test = le.transfrm(y_test)

    print(datetime.datetime.now(), 'Training CGAN')
    cigan = CGAN(img_rows, img_cols, channels, n_classes, latent_dim,
                 img_path=path_cgan_images+'/imgs/%s_' % dataset, verbose=True)
    ts = time.time()
    cigan.fit(X_train, y_train, epochs=epochs, batch_size=32, sample_interval=100)
    cgan_fit_time = time.time() - ts

    n_fake_instances = len(X_train)

    print(datetime.datetime.now(), 'Generating synthetic data')
    ts = time.time()
    X_fake = cigan.sample(n_fake_instances)
    cgan_gen_time = time.time() - ts

    print(datetime.datetime.now(), 'Storing synthetic data')
    np.save(path_syht_dataset + '%s' % dataset, X_fake)

    s0, s1, s2 = X_fake.shape
    X_fake = X_fake.reshape(s0, s1 * s2)
    X_real = X_train.reshape(s0, s1 * s2)

    # print(np.mean(X_fake[0]), np.min(X_fake[0]), np.max(X_fake[0]))
    # print(np.mean(X_real[0]), np.min(X_real[0]), np.max(X_real[0]))

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
    store_result(res_dict, path_ctgan_eval + 'images.json')


if __name__ == "__main__":
    main()
