from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Embedding
from keras.layers.advanced_activations import LeakyReLU


def clean_text(x):
    stop_words = set(stopwords.words('english'))
    x1 = x.translate(str.maketrans('', '', string.punctuation))
    word_tokens = word_tokenize(x1)
    x2 = [w.lower() for w in word_tokens if not w.lower() in stop_words]
    x3 = ' '.join(x2)
    return x3


# s = 'This is a sample sentence, showing off the stop words filtration.'
# print(clean_text(s))

def clean_texts(X, min_chars=0):
    X1 = [clean_text(x) for x in X]
    X2 = [x for x in X1 if len(x) > min_chars]
    return X2

#
# # path = get_file(
# #     'nietzsche.txt',
# #     origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
# # with io.open(path, encoding='utf-8') as f:
# #     text = f.read().lower()
# # print('corpus length:', len(text))


# def texts2chars(X):
#     texts = ' '.join(X)
#     chars = sorted(list(set(texts)))
#     char_indices = dict((c, i) for i, c in enumerate(chars))
#     indices_char = dict((i, c) for i, c in enumerate(chars))
#     return chars, char_indices, indices_char
#
#
# # cut the text in semi-redundant sequences of maxlen characters
# def texts2texts_nextchar(texts, maxlen=100, step=3):
#
#     chars, char_indices, indices_char = texts2chars(texts)
#
#     sentences = list()
#     next_chars = list()
#     for text in texts:
#         for i in range(0, len(text) - maxlen, step):
#             sentences.append(text[i: i + maxlen])
#             next_chars.append(text[i + maxlen])
#
#     X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
#     y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
#     for i, sentence in enumerate(sentences):
#         for t, char in enumerate(sentence):
#             X[i, t, char_indices[char]] = 1
#         y[i, char_indices[next_chars[i]]] = 1


def texts2words(X):
    texts = ' '.join(X)
    words = sorted(word_tokenize(texts))
    words_indices = dict((c, i) for i, c in enumerate(words))
    indices_words = dict((i, c) for i, c in enumerate(words))
    return words, words_indices, indices_words


# cut the text in semi-redundant sequences of maxlen words
def texts2texts_nextword(texts, maxlen=100, step=3):

    words, words_indices, indices_words = texts2words(texts)

    sentences = list()
    next_word = list()
    for text in texts:
        word_tokens = word_tokenize(text)
        for i in range(0, len(word_tokens) - maxlen, step):
            sentences.append(word_tokens[i: i + maxlen])
            next_word.append(word_tokens[i + maxlen])

    X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
    y = np.zeros((len(sentences), len(words)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, words_indices[char]] = 1
        y[i, words_indices[next_word[i]]] = 1

    return X, y, words, words_indices, indices_words


def sample_pred(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class GAN:
    def __init__(self, nbr_terms, maxlen, latent_dim, term_indices, indices_term,
                 txt_path=None, verbose=False, start_texts=None, text_lengths=None):

        self.nbr_terms = nbr_terms
        self.maxlen = maxlen
        self.txt_shape = (maxlen, nbr_terms)
        self.latent_dim = latent_dim
        self.term_indices = term_indices
        self.indices_term = indices_term
        self.txt_path = txt_path
        self.verbose = verbose
        self.start_texts = start_texts
        self.text_lengths = text_lengths

        self.model = Sequential()
        self.model.add(LSTM(self.latent_dim, input_shape=(maxlen, self.nbr_terms)))
        self.model.add(Dense(self.nbr_terms, activation='softmax'))

        optimizer = RMSprop(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        self.print_callback = LambdaCallback(on_epoch_end=self.store_texts)

    def store_texts(self, epoch, _):

        t = 20
        if epoch % 10 == 0 and self.txt_path:
            fout = open(self.txt_path + 'cgan_txt_%d.txt' % epoch, 'w')

            idx = np.random.choice(len(self.start_texts))
            text = word_tokenize(self.start_texts[idx])

            if self.maxlen + 1 < len(text):
                start_index = random.randint(0, len(text) - self.maxlen - 1)
            else:
                start_index = 0

            for diversity in [0.2, 0.5, 1.0, 1.2]:
                fout.write(str(diversity) + '-')

                generated = list()
                sentence = text[start_index: start_index + self.maxlen]
                generated += sentence
                # print(-1, ' '.join(sentence))

                if self.text_lengths is not None:
                    tr = int(np.round(np.random.normal(
                        loc=np.mean(self.text_lengths), scale=np.std(self.text_lengths))))
                    t0 = max(np.min(self.text_lengths), tr)
                else:
                    t0 = t

                for i in range(t0):
                    x_pred = np.zeros((1, self.maxlen, self.nbr_terms))
                    for t, term in enumerate(sentence):
                        x_pred[0, t, self.term_indices[term]] = 1.

                    preds = self.model.predict(x_pred, verbose=0)[0]
                    next_index = sample_pred(preds, diversity)
                    next_term = self.indices_term[next_index]

                    sentence = sentence[1:] + [next_term]
                    generated += [next_term]
                    # print(i, ' '.join(sentence))

                if self.verbose:
                    print(str(diversity), ' '.join(generated))

                fout.write(' '.join(generated))
                fout.write('\n')
                fout.flush()

            fout.close()

    def fit(self, X, y, epochs, batch_size=128):

        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[self.print_callback])

    def sample(self, n, t, start_texts, diversity_list):
        samples = list()
        for i in range(n):
            diversity = np.random.choice(diversity_list)
            idx = np.random.choice(len(start_texts))
            text = word_tokenize(start_texts[idx])

            if self.maxlen + 1 < len(text):
                start_index = random.randint(0, len(text) - self.maxlen - 1)
            else:
                start_index = 0

            generated = list()
            sentence = text[start_index: start_index + self.maxlen]
            generated += sentence

            if self.text_lengths is not None:
                tr = int(np.round(np.random.normal(
                    loc=np.mean(self.text_lengths), scale=np.std(self.text_lengths))))
                t0 = max(np.min(self.text_lengths), tr)
            else:
                t0 = t

            for j in range(t0):
                x_pred = np.zeros((1, self.maxlen, self.nbr_terms))
                for t, term in enumerate(sentence):
                    x_pred[0, t, self.term_indices[term]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample_pred(preds, diversity)
                next_term = self.indices_term[next_index]

                sentence = sentence[1:] + [next_term]
                generated += [next_term]

            samples.append(' '.join(generated))

        return samples

