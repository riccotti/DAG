import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam


class CGAN:
    def __init__(self, img_rows, img_cols, channels, n_classes, latent_dim, img_path=None, verbose=False):
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = n_classes
        self.latent_dim = latent_dim
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
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

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

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def fit(self, X, y, epochs, batch_size=128, sample_interval=100):

        X_train = np.expand_dims(X, axis=3)
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
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
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
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if self.img_path is not None and epoch % sample_interval == 0:
                self.store_images(epoch)

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

        gen_imgs = self.generator.predict([noise, sampled_labels])
        s0, s1, s2, _ = gen_imgs.shape
        gen_imgs = gen_imgs.reshape((s0, s1, s2))

        if return_labels:
            return gen_imgs, sampled_labels

        return gen_imgs

    def store_images(self, epoch):
        r, c = 2, 5

        gen_imgs, sampled_labels = self.sample(r * c, return_labels=True)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
                axs[i, j].set_title("class %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(self.img_path + 'cgan_img_%d.png' % epoch)
        plt.close()


# if __name__ == '__main__':
#     # Load the dataset
#     # (X_train, _), (X_test, _) = mnist.load_data()
#
#     dataset = 'mnist'
#     train_size = 1000
#
#     D = get_dataset(dataset, path_dataset, categories=[0, 1, 2, 3, 4, 5, 6, 7, 8])
#
#     X_train = D['X_train']
#     X_test = D['X_test']
#     y_train = D['y_train']
#     y_test = D['y_train']
#     n_classes = D['n_classes']
#     print('--->', n_classes, np.unique(y_train))
#
#     # Rescale -1 to 1
#     # X_train = X_train / 127.5 - 1.
#     print(X_train.shape)
#     X_train = np.expand_dims(X_train, axis=3)
#     print(X_train.shape)
#
#     # img_rows, img_cols = X_train.shape
#     img_rows, img_cols = D['w'], D['h']
#     channels = 1
#     latent_dim = 100
#     gan = CGAN(img_rows, img_cols, channels, n_classes, latent_dim)
#     gan.train(X_train, y_train, epochs=100, batch_size=32, sample_interval=100)
#
#     plt.imshow(X_test[0] * 255, cmap='gray')
#     plt.show()
#
#     nbr_da_generare = 1
#     sampled_labels = y_test[:nbr_da_generare]
#     A = gan.generator.predict([np.random.normal(0, 1, (nbr_da_generare, latent_dim)), sampled_labels])
#     print(A.shape)
#     # print(A)
#
#     plt.imshow(A[0].reshape(28, 28) * 255, cmap='gray')
#     plt.show()
