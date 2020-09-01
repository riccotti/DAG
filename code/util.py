import os
import pickle
import imageio
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from nltk.corpus import stopwords
from skimage.color import rgb2gray

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from pyts.preprocessing import MinMaxScaler as TsMinMaxScaler, StandardScaler as TsStandardScaler

from skimage import filters

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from tslearn.datasets import UCR_UEA_datasets
from sklearn.datasets import fetch_20newsgroups, make_classification
from keras.datasets import mnist, cifar10, cifar100, fashion_mnist, imdb

datasets = {
    'avila': ('avila.csv', 'tab'),
    'ctg': ('ctg.csv', 'tab'),
    'diabetes': ('diabetes.csv', 'tab'),
    'ionoshpere': ('ionosphere.csv', 'tab'),
    'mouse': ('mouse.csv', 'tab'),
    'parkinsons': ('parkinsons.csv', 'tab'),
    'sonar': ('sonar.csv', 'tab'),
    'vehicle': ('vehicle.csv', 'tab'),
    'wdbc': ('wdbc.csv', 'tab'),
    'wine': ('wine.csv', 'tab'),
    'winer': ('wine-red.csv', 'tab'),
    'winew': ('wine-white.csv', 'tab'),
    'rnd': ('', 'rnd'),

    'mnist': ('', 'img'),
    'fashion_mnist': ('', 'img'),
    'cifar10': ('', 'img'),
    'cifar100': ('', 'img'),
    'omniglot': ('omniglot', 'img'),

    'gunpoint': ('GunPoint', 'ts'),
    'italypower': ('ItalyPowerDemand', 'ts'),
    'arrowhead': ('ArrowHead', 'ts'),
    'ecg200': ('ECG200', 'ts'),
    'ecg5000': ('ECG5000', 'ts'),
    'electricdevices': ('ElectricDevices', 'ts'),
    'phalanges': ('PhalangesOutlinesCorrect', 'ts'),

    '20newsgroups': ('', 'txt'),
    'imdb': ('', 'txt'),

}


def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name].fillna(mode, inplace=True)
    return df


def get_avila_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df = df[df['class'] != 'B']
    df = df[df['class'] != 'W']
    return df, class_name


def get_ctg_dataset(filename):
    class_name = 'CLASS'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df.drop(['FileName', 'Date', 'SegFile', 'NSP'], axis=1, inplace=True)
    return df, class_name


def get_diabetes_dataset(filename):
    class_name = 'Outcome'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_ionosphere_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_mouse_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df.drop(['MouseID', 'Genotype', 'Treatment', 'Behavior'], axis=1, inplace=True)
    return df, class_name


def get_mnist_dataset(filename=None, categories=None):
    w, h = 28, 28
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], w, h)
    X_test = X_test.reshape(X_test.shape[0], w, h)

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    if categories is not None:
        idx_train = np.array([np.where(c == y_train)[0] for c in categories])
        idx_train = np.concatenate(idx_train)
        idx_test = np.array([np.where(c == y_test)[0] for c in categories])
        idx_test = np.concatenate(idx_test)
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test = X_test[idx_test], y_test[idx_test]

    window_sizes = [(7, 7), (14, 14), (14, 14)]
    window_steps = [(7, 7), (7, 7), (14, 14)]

    return X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps


def get_fashion_mnist_dataset(filename=None, categories=None):
    w, h = 28, 28
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], w, h)
    X_test = X_test.reshape(X_test.shape[0], w, h)

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    if categories is not None:
        idx_train = np.array([np.where(c == y_train)[0] for c in categories])
        idx_train = np.concatenate(idx_train)
        idx_test = np.array([np.where(c == y_test)[0] for c in categories])
        idx_test = np.concatenate(idx_test)
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test = X_test[idx_test], y_test[idx_test]

    window_sizes = [(7, 7), (14, 14), (14, 14)]
    window_steps = [(7, 7), (7, 7), (14, 14)]

    return X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps


# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_cifar10_dataset(filename=None, categories=None):
    w, h = 32, 32
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = np.array([rgb2gray(x) for x in X_train])
    X_test = np.array([rgb2gray(x) for x in X_test])

    X_train = X_train.reshape((X_train.shape[0], w, h))
    X_test = X_test.reshape((X_test.shape[0], w, h))

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    if categories is not None:
        idx_train = np.array([np.where(c == y_train)[0] for c in categories])
        idx_train = np.concatenate(idx_train)
        idx_test = np.array([np.where(c == y_test)[0] for c in categories])
        idx_test = np.concatenate(idx_test)
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test = X_test[idx_test], y_test[idx_test]

    window_sizes = [(8, 8), (16, 16), (16, 16)]
    window_steps = [(8, 8), (8, 8), (16, 16)]

    return X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps


def get_cifar100_dataset(filename=None, categories=None):
    w, h = 32, 32
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    X_train = np.array([rgb2gray(x) for x in X_train])
    X_test = np.array([rgb2gray(x) for x in X_test])

    X_train = X_train.reshape((X_train.shape[0], w, h))
    X_test = X_test.reshape((X_test.shape[0], w, h))

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    if categories is not None:
        idx_train = np.where(categories == y_train)[0]
        idx_test = np.where(categories == y_test)[0]
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test = X_test[idx_test], y_test[idx_test]

    window_sizes = [(8, 8), (16, 16), (16, 16)]
    window_steps = [(8, 8), (8, 8), (16, 16)]

    return X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps


def _get_omniglot_dataset(path, verbose=False):
    X = list()
    y = list()
    cat_dict = dict()
    lang_dict = dict()
    curr_y = 0
    # we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):
        if alphabet.startswith('.'):
            continue
        if verbose:
            print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)
        # every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(alphabet_path):
            if letter.startswith('.'):
                continue
            cat_dict[curr_y] = (alphabet, letter)
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imageio.imread(image_path)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X, y, lang_dict


def get_omniglot_dataset(path):

    train_pickle_filename = path + '/omniglot_train.pickle'
    if not os.path.exists(train_pickle_filename):
        X, y, c = _get_omniglot_dataset(path + '/images_background', verbose=False)

        with open(train_pickle_filename, 'wb') as f:
            pickle.dump((X, y, c), f)

    test_pickle_filename = path + '/omniglot_test.pickle'
    if not os.path.exists(test_pickle_filename):
        X, y, c = _get_omniglot_dataset(path + '/images_evaluation', verbose=False)

        with open(test_pickle_filename, 'wb') as f:
            pickle.dump((X, y, c), f)

    with open(train_pickle_filename, 'rb') as f:
        (X_train, y_train, classes_train) = pickle.load(f)

    with open(test_pickle_filename, 'rb') as f:
        (X_test, y_test, classes_test) = pickle.load(f)

    w, h = 105, 105

    return X_train, X_test, y_train, y_test, w, h


def get_parkinsons_dataset(filename):
    class_name = 'status'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df.drop(['name'], axis=1, inplace=True)
    return df, class_name


def get_wdbc_dataset(filename):
    class_name = 'diagnosis'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    df.drop(['id'], axis=1, inplace=True)
    return df, class_name


def get_sonar_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_vehicle_dataset(filename):
    class_name = 'CLASS'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_wine_dataset(filename):
    class_name = 'quality'
    df = pd.read_csv(filename, sep=',', skipinitialspace=True)
    return df, class_name


def get_arrowhead_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ArrowHead')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_gunpoint_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('GunPoint')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_ecg200_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ECG200')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_ecg5000_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ECG5000')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_italypower_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ItalyPowerDemand')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [2, 4, 6]
    window_steps = [1, 1, 1]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_electricdevices_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('ElectricDevices')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_phalanges_dataset(filename):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('PhalangesOutlinesCorrect')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    n_timestamps = X_train.shape[1]
    window_sizes = [4, 8, 16]
    window_steps = [1, 2, 4]
    return X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps


def get_20news_dataset(filename, n_words=1000, encoding='tfidf', categories=None):

    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                                          categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                                         categories=categories)

    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    X_train_txt, X_test_txt = X_train, X_test

    if encoding == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=n_words, stop_words=stopwords.words('english'))
        X_train = vectorizer.fit_transform(X_train).toarray()
        X_test = vectorizer.transform(X_test).toarray()
        return X_train, X_test, y_train, y_test, n_words, vectorizer, X_train_txt, X_test_txt
    elif encoding == 'cv':
        vectorizer = CountVectorizer(max_features=n_words, stop_words=stopwords.words('english'))
        X_train = vectorizer.fit_transform(X_train).toarray()
        X_test = vectorizer.transform(X_test).toarray()
        return X_train, X_test, y_train, y_test, n_words, vectorizer, X_train_txt, X_test_txt
    elif encoding == 'wordembedding':
        print('To implement!')
        return None

    return None


# index = imdb.get_word_index()
# reverse_index = dict([(value, key) for (key, value) in index.items()])
# decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] )

def get_imdb_dataset(filename, n_words=1000, encoding='tfidf', categories=None):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(path='imdb.npz', num_words=None, skip_top=0, maxlen=None,
                                                          seed=113, start_char=1, oov_char=2, index_from=3)
    index = imdb.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in index.items()])
    X_train_txt = [' '.join([reverse_index.get(i - 3, "#") for i in xi]) for xi in X_train]
    X_test_txt = [' '.join([reverse_index.get(i - 3, "#") for i in xi]) for xi in X_test]

    if encoding == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=n_words, stop_words=stopwords.words('english'))
        X_train = vectorizer.fit_transform(X_train_txt).toarray()
        X_test = vectorizer.transform(X_test_txt).toarray()
        return X_train, X_test, y_train, y_test, n_words, vectorizer, X_train_txt, X_test_txt
    elif encoding == 'cv':
        vectorizer = CountVectorizer(max_features=n_words, stop_words=stopwords.words('english'))
        X_train = vectorizer.fit_transform(X_train).toarray()
        X_test = vectorizer.transform(X_test).toarray()
        return X_train, X_test, y_train, y_test, n_words, vectorizer, X_train_txt, X_test_txt
    elif encoding == 'wordembedding':
        print('To implement!')
        return None

    return None


def get_random_dataset(n_samples, n_features, n_informative, n_classes, test_size, random_state):
    class_name = 'class'
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_classes=n_classes,
                               random_state=random_state,
                               shuffle=False)
    columns = [str(c) for c in range(n_features)] + [class_name]
    df = pd.DataFrame(data=np.concatenate([X, y.reshape(len(y), 1)], axis=1), columns=columns)

    feature_names = [c for c in df.columns if c != class_name]
    class_values = sorted(np.unique(df[class_name]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    dataset = {
        'name': 'rnd',
        'data_type': 'rnd',
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'feature_names': feature_names,
        'n_features': n_features,
        'n_classes': n_classes,
        'n_samples': n_samples,
        'n_informative': n_informative
    }

    return dataset


dataset_read_function_map = {
    '20newsgroups': get_20news_dataset,
    'arrowhead': get_arrowhead_dataset,
    'avila': get_avila_dataset,
    'ctg': get_ctg_dataset,
    'cifar10': get_cifar10_dataset,
    'cifar100': get_cifar100_dataset,
    'diabetes': get_diabetes_dataset,
    'ecg200': get_ecg200_dataset,
    'ecg5000': get_ecg5000_dataset,
    'fashion_mnist': get_fashion_mnist_dataset,
    'gunpoint': get_gunpoint_dataset,
    'ionoshpere': get_ionosphere_dataset,
    'imdb': get_imdb_dataset,
    'italypower': get_italypower_dataset,
    'mouse': get_mouse_dataset,
    'mnist': get_mnist_dataset,
    'omniglot': get_omniglot_dataset,
    'parkinsons': get_parkinsons_dataset,
    'sonar': get_sonar_dataset,
    'vehicle': get_vehicle_dataset,
    'wdbc': get_wdbc_dataset,
    'wine': get_wine_dataset,
    'winer': get_wine_dataset,
    'winew': get_wine_dataset,
    'rnd': get_random_dataset,

    'electricdevices': get_electricdevices_dataset,
    'phalanges': get_phalanges_dataset,
}


def get_tabular_dataset(name, path='./', normalize=None, test_size=0.3, random_state=None):

    get_dataset_fn = dataset_read_function_map[name]

    filename = path + datasets[name][0]
    data_type = datasets[name][1]

    df, class_name = get_dataset_fn(filename)

    feature_names = [c for c in df.columns if c != class_name]
    class_values = sorted(np.unique(df[class_name]))

    X = df[feature_names].values
    y = df[class_name].values

    if normalize == 'minmax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif normalize == 'standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)
    n_classes = len(class_values)
    n_features = len(feature_names)

    dataset = {
        'name': name,
        'data_type': data_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'feature_names': feature_names,
        'n_features': n_features,
        'n_classes': n_classes,
    }

    return dataset


def get_image_dataset(name, path, categories, filter):

    get_dataset_fn = dataset_read_function_map[name]

    filename = path + datasets[name][0]
    data_type = datasets[name][1]

    X_train, X_test, y_train, y_test, w, h, window_sizes, window_steps = get_dataset_fn(filename, categories)

    if filter == 'sobel':
        X_train = np.array([filters.sobel(x) for x in X_train])
        X_test = np.array([filters.sobel(x) for x in X_test])
    elif filter == 'roberts':
        X_train = np.array([filters.roberts(x) for x in X_train])
        X_test = np.array([filters.roberts(x) for x in X_test])

    class_name = 'class'
    class_values = sorted(np.unique(y_train))
    n_classes = len(class_values)

    dataset = {
        'name': name,
        'data_type': data_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'w': w,
        'h': h,
        'n_classes': n_classes,
        'window_sizes': window_sizes,
        'window_steps': window_steps,
    }

    return dataset


def get_ts_dataset(name, path, normalize=None):
    get_dataset_fn = dataset_read_function_map[name]

    filename = path + datasets[name][0]
    data_type = datasets[name][1]

    X_train, X_test, y_train, y_test, n_timestamps, window_sizes, window_steps = get_dataset_fn(filename)

    if normalize == 'minmax':
        scaler = TsMinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif normalize == 'standard':
        scaler = TsStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    class_name = 'class'
    class_values = sorted(np.unique(y_train))
    n_classes = len(class_values)

    dataset = {
        'name': name,
        'data_type': data_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'n_timestamps': n_timestamps,
        'n_classes': n_classes,
        'window_sizes': window_sizes,
        'window_steps': window_steps
    }

    return dataset


def get_txt_dataset(name, path, n_words, encoding, categories):
    get_dataset_fn = dataset_read_function_map[name]

    filename = path + datasets[name][0]
    data_type = datasets[name][1]

    X_train, X_test, y_train, y_test, n_terms, vectorizer, X_train_txt, X_test_txt = get_dataset_fn(filename, n_words, encoding, categories)

    class_name = 'class'
    class_values = sorted(np.unique(y_train))
    n_classes = len(class_values)

    dataset = {
        'name': name,
        'data_type': data_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'class_name': class_name,
        'class_values': class_values,
        'n_terms': n_terms,
        'n_classes': n_classes,
        'vectorizer': vectorizer,
        'X_train_txt': X_train_txt,
        'X_test_txt': X_test_txt,
    }

    return dataset


def get_dataset(name, path='./', normalize=None, test_size=0.3, random_state=None, **kwargs):

    if name not in datasets:
        raise ValueError('Unknown dataset %s' % name)

    dataset_type = datasets[name][1]

    if dataset_type == 'tab':
        return get_tabular_dataset(name, path, normalize, test_size, random_state)
    elif dataset_type == 'img':
        categories = kwargs.get('categories', None)
        filter = kwargs.get('filter', None)
        return get_image_dataset(name, path, categories, filter)
    elif dataset_type == 'ts':
        return get_ts_dataset(name, path, normalize)
    elif dataset_type == 'txt':
        n_words = kwargs.get('n_words', 1000)
        encoding = kwargs.get('encoding', 'tfidf')
        categories = kwargs.get('categories', None)
        return get_txt_dataset(name, path, n_words, encoding, categories)
    elif dataset_type == 'rnd':
        min_samples = kwargs.get('min_samples', 10000)
        max_samples = kwargs.get('max_samples', 100000)
        min_features = kwargs.get('min_features', 10)
        max_features = kwargs.get('max_features', 1000)
        min_classes = kwargs.get('min_classes', 2)
        max_classes = kwargs.get('max_classes', 20)
        n_samples = np.random.randint(min_samples, max_samples + 1)
        n_features = np.random.randint(min_features, max_features + 1)
        n_informative = np.random.randint(0, n_features // 2)
        n_classes = np.random.randint(min_classes, max_classes + 1)
        return get_random_dataset(n_samples, n_features, n_informative, n_classes, test_size, random_state)


def fsl_data_format(X, y, sample_size=None, random_state=None):
    X, y = shuffle(X, y, random_state=random_state)

    X1 = defaultdict(list)
    for xi, yi in zip(X, y):
        X1[yi].append(xi)

    sizes = [len(X1[k]) for k in X1]
    if sample_size is None:
        sample_size = min(sizes)
    else:
        sample_size = min(min(sizes), sample_size)

    X2 = list()
    y2 = list()
    for k in X1:
        X2.append(np.stack(X1[k][:sample_size]))
        y2.append([k] * sample_size)

    X2 = np.stack(X2)
    y2 = np.stack(y2)

    return X2, y2


def split_train_test(X, y, test_size=0.3, random_state=None):
    X, y = shuffle(X, y, random_state=random_state)

    split_index = int(np.ceil(len(X[0]) * (1.0 - test_size) / 2) * 2)

    X_train, X_test = X[:, :split_index, :], X[:, split_index:, :]
    y_train, y_test = y[:, :split_index], y[:, split_index:]

    return X_train, X_test, y_train, y_test


