import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Lambda
from keras.layers import BatchNormalization, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam


def moving_average(ts, window=3):
    ma = pd.Series(ts).rolling(window=window).mean()
    for i in range(window - 1):
        ma[i] = ma[window - 1]
    return ma


class CGAN:
    def __init__(self, n_timestamps, n_classes, latent_dim, window=None, img_path=None, verbose=False):
        # Input shape
        self.n_timestamps = n_timestamps
        self.img_shape = (self.n_timestamps,)
        self.num_classes = n_classes
        self.latent_dim = latent_dim
        self.window = window
        self.img_path = img_path
        self.verbose = verbose

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        ts = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([ts, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        if self.verbose:
            model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        if self.verbose:
            model.summary()

        ts = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        # flat_img = Flatten()(img)
        flat_img = Lambda(lambda x: x)(ts)

        model_input = multiply([flat_img, label_embedding])
        validity = model(model_input)

        return Model([ts, label], validity)

    def fit(self, X, y, epochs, batch_size=128, sample_interval=50):

        X_train = X
        y_train = y

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            tss, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_tss = self.generator.predict([noise, labels])
            # if self.window is not None:
            #     gen_tss = np.array([moving_average(gts, self.window) for gts in gen_tss])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([tss, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_tss, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            if self.verbose and epoch % sample_interval == 0:
                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if self.img_path is not None and epoch % sample_interval == 0:
                self.store_images(epoch)

    def sample(self, n, labels=None, return_labels=False):
        noise = np.random.normal(0, 1, (n, self.latent_dim))
        if labels is None:
            sampled_labels = np.random.choice(self.num_classes, size=n, replace=True).reshape(-1, 1)
        else:
            instances_per_labels = max(1, n // len(labels))
            sampled_labels = np.concatenate([[l] * instances_per_labels for l in labels])

        gen_ts = self.generator.predict([noise, sampled_labels])
        if self.window is not None:
            gen_ts = np.array([moving_average(gts, self.window) for gts in gen_ts])

        if return_labels:
            return gen_ts, sampled_labels

        return gen_ts

    def store_images(self, epoch):
        r, c = 2, 5

        gen_ts, sampled_labels = self.sample(r * c, return_labels=True)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].plot(gen_ts[cnt])
                axs[i, j].set_title("class %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(self.img_path + 'cgan_ts_%d.png' % epoch)
        plt.close()


# if __name__ == '__main__':
#     dataset = 'italypower'
#     train_size = 1000
#
#     D = get_dataset(dataset, path_dataset, normalize='standard')
#     X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
#     n_timestamps = D['n_timestamps']
#     y_train = y_train - 1
#     y_test = y_test - 1
#
#     n_classes = D['n_classes']
#
#     # X_train = np.expand_dims(X_train, axis=2)
#
#     latent_dim = 10
#     gan = CGAN(n_timestamps, n_classes, latent_dim)
#     gan.train(X_train, y_train, epochs=1000, batch_size=32, sample_interval=100)
#
#     plt.plot(X_test[0])
#     plt.show()
#
#     nbr_da_generare = 1
#     sampled_labels = y_test[:nbr_da_generare]
#     A = gan.generator.predict([np.random.normal(0, 1, (nbr_da_generare, latent_dim)), sampled_labels])
#     print(A.shape)
#     # print(A)
#
#     plt.plot(A[0])
#     plt.show()
