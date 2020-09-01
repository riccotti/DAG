
import numpy as np

from copy import copy, deepcopy
from abc import ABC, abstractmethod

from skimage.util.shape import view_as_windows
from nltk.tokenize import regexp_tokenize
from keras.preprocessing.text import text_to_word_sequence


def euclidean_dist(a, b):
    return np.sqrt(np.sum((a - b)**2))


def absolute_diff(a, b):
    return np.sum(np.abs(a - b))


def _distance(x, s, dist, min_dist=None):
    d = 0.0
    for i in range(len(x)):
        d += dist(x[i], s[i])
        if min_dist and d > min_dist:
            return d
    return d


def distance(x, s, dist=None):
    min_dist = np.inf
    dist = dist if dist is not None else euclidean_dist
    for i in range(len(x)):
        d = _distance(x[i], s, dist, min_dist)
        if d < min_dist:
            min_dist = d

    return min_dist


class DataRecord(ABC):

    def __init__(self, data):
        self.data = data
        self.shape = self.data.shape
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __setitem__(self, index, item):
        pass

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v))
        return result

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)


class TabularRecord(DataRecord):

    def __init__(self, data):
        super().__init__(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, item):
        self.data[index] = item


class ImageRecord(DataRecord):

    def __init__(self, data, window_shape=(1, 1), step=1):
        self.window_shape = window_shape
        self.step = step
        self.view = view_as_windows(data, window_shape=window_shape, step=step)
        self.length = self.view.shape[0] * self.view.shape[1]
        self.view = self.view.reshape(self.length, self.view.shape[2], self.view.shape[3])

        super().__init__(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.view[index]

    def __setitem__(self, index, item):
        self.view[index] = item
        w, h = self.data.shape
        div_w = (w - self.window_shape[0]) // self.step + 1
        div_h = (h - self.window_shape[1]) // self.step + 1
        for idx, v in enumerate(self.view):
            j = (idx // div_w) * self.step
            i = (idx % div_h) * self.step
            self.data[j:j + self.window_shape[1], i:i + self.window_shape[0]] = v
        self.view = view_as_windows(self.data, window_shape=self.window_shape, step=self.step)
        self.view = self.view.reshape(self.length, self.view.shape[2], self.view.shape[3])


class TimeSeriesRecord(DataRecord):

    def __init__(self, data, window_shape=1, step=1):

        self.window_shape = window_shape
        self.step = step
        self.view = view_as_windows(data, window_shape=window_shape, step=step)
        self.length = self.view.shape[0]

        super().__init__(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.view[index]

    def __setitem__(self, index, item):
        self.view[index] = item
        div = (len(self.data) - self.window_shape) // self.step + 1
        for idx, v in enumerate(self.view):
            i = (idx // div) * self.step
            self.data[i:i + self.window_shape] = v
        self.view = view_as_windows(self.data, window_shape=self.window_shape, step=self.step)


def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    punc_list = '!"#$%&()*+,-./:;<=>?@[/]^_{|}~' + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.translate(t)

    t = str.maketrans(dict.fromkeys("'`", " "))
    text = text.translate(t)

    return text


def renltk_tokenize(text):
    text = clean_text(text)
    words = regexp_tokenize(text, pattern='\s+', gaps=True)
    return words


def keras_tokenize(text):
    text = clean_text(text)
    words = text_to_word_sequence(text)
    return words


class TextRecord(TimeSeriesRecord):

    def __init__(self, data, window_shape=1, step=1, text_length=10000):
        data = np.array(renltk_tokenize(data))
        # if len(data) <= window_shape:
        #     data = 'This is a dummy sentence.'
        #     data = np.array(renltk_tokenize(data))

        if len(data) > text_length:
            data = data[:text_length]
        else:
            pad = np.array([''] * (text_length - len(data)))
            data = np.concatenate([data, pad])

        super().__init__(data, window_shape, step)
