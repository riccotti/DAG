import time
import json
import pickle
import datetime
import numpy as np

# from textgenrnn import textgenrnn

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


from dang.config import *
from dang.util import get_dataset

from nltk.corpus import stopwords
from dang.txt_gan import GAN, clean_texts, texts2texts_nextword

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

    dataset = 'imdb'
    epochs = 100
    train_size = 0.7
    maxlen = 3
    step = 1

    print(datetime.datetime.now(), 'Dataset: %s' % dataset)
    if dataset == '20newsgroups':
        categories = ['alt.atheism', 'talk.religion.misc']
    else:
        categories = None
    D = get_dataset(dataset, path_dataset, categories=categories)

    # X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    # n_classes = D['n_classes']
    # class_name = D['class_name']
    X_train_txt = D['X_train_txt']
    y_train = D['y_train']
    if dataset == 'imdb':
        X_train_txt, _, y_train, _ = train_test_split(X_train_txt, y_train,
                                                                 train_size=1000,
                                                                 stratify=y_train)

    X_train_txt = clean_texts(X_train_txt, min_chars=10)
    text_lengths = [len(x) for x in X_train_txt]

    # X_test_txt = D['X_test_txt']
    # X_test_txt = clean_texts(X_test_txt, min_chars=10)

    # print(np.mean(lens), np.median(lens), np.min(lens), np.max(lens))
    # return -1

    X, y, words, words_indices, indices_words = texts2texts_nextword(X_train_txt, maxlen=maxlen, step=step)
    nbr_terms = len(words)
    start_texts_idx = np.random.choice(len(X_train_txt), min(1000, len(X_train_txt)), replace=False)
    start_texts = [X_train_txt[i] for i in start_texts_idx]

    print(datetime.datetime.now(), 'Training GAN')
    gan = GAN(nbr_terms, maxlen, latent_dim=100, term_indices=words_indices, indices_term=indices_words,
              txt_path=path_cgan_images+'/txt/%s_' % dataset, verbose=False,
              start_texts=start_texts, text_lengths=text_lengths)
    ts = time.time()
    gan.fit(X, y, epochs=epochs, batch_size=128)
    cgan_fit_time = time.time() - ts

    n_fake_instances = len(X_train_txt)

    print(datetime.datetime.now(), 'Generating synthetic data')
    ts = time.time()
    X_fake_txt = gan.sample(n=n_fake_instances, t=20, start_texts=start_texts,
                            diversity_list=[0.2, 0.5, 0.7, 1.0, 1.2])
    cgan_gen_time = time.time() - ts

    # print('Fake start')
    # for f in X_fake_txt[:5]:
    #     print(f)
    # print('Fake end')
    #
    # print('Real start')
    # for r in X_train_txt[:5]:
    #     print(r)
    # print('Real end')
    # # return -1

    print(datetime.datetime.now(), 'Storing synthetic data')
    # df = pd.DataFrame(data=Xy_fake, columns=feature_names + [class_name])
    # df.to_csv(path_syht_dataset + '%s.csv' % dataset, index=False)
    fout = open(path_syht_dataset + '%s.csv' % dataset, 'w')
    for txt in X_fake_txt:
        fout.write(txt + '\n')
    fout.close()

    X_fake = X_fake_txt
    X_real = X_train_txt

    y_real = np.ones(len(X_real))
    y_fake = np.zeros(len(X_fake))

    X_rf = np.concatenate([X_real, X_fake])
    y_rf = np.concatenate([y_real, y_fake])

    X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, train_size=train_size, stratify=y_rf)

    vectorizer = TfidfVectorizer(max_features=1000, stop_words=stopwords.words('english'))
    X_rf_train = vectorizer.fit_transform(X_rf_train).toarray()
    X_rf_test = vectorizer.transform(X_rf_test).toarray()

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
    store_result(res_dict, path_ctgan_eval + 'text.json')


if __name__ == "__main__":
    main()
