from pathlib import Path

import cv2
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import generic_utils

from easy_inference.train_utils.batching import batch_iterator

from model_prep.infogan.generator import GeneratorDeconv
from model_prep.infogan.discriminator import Discriminator


class Trainer:
    def __init__(self, generator, discriminator, save_dir):
        self.generator: GeneratorDeconv = generator
        self.discriminator: Discriminator = discriminator
        self.save_dir = Path(save_dir).resolve()

        # Create necessary directories
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _setup_models(self):
        """
        Set up models for training
        """

        # Create the DCGAN model
        gan = Model(inputs=self.generator.model.inputs,
                    outputs=self.discriminator.model(
                        self.generator.model(self.generator.model.inputs)),
                    name="DCGAN")

        # Create Optimizers
        opt_discriminator = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999,
                                 epsilon=1e-08)
        opt_dcgan = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

        # Compile each model
        self.generator.model.trainable = True
        self.generator.model.compile(loss='mse',
                                     optimizer=opt_discriminator)

        self.discriminator.model.trainable = False
        losses = ['binary_crossentropy', 'categorical_crossentropy',
                  gaussian_loss]
        gan.compile(loss=losses, loss_weights=[1, 1, 1], optimizer=opt_dcgan)
        gan.summary()

        self.discriminator.model.trainable = True
        self.discriminator.model.compile(loss=losses,
                                         loss_weights=[1, 1, 1],
                                         optimizer=opt_discriminator)
        return gan

    def _train_epoch(self, x_train, gan, batch_size, noise_scale):
        # Create a progress bar for the Epoch
        progbar = generic_utils.Progbar(x_train.shape[0])

        # Create a batch iterator
        batches = batch_iterator(x_train, batch_size)

        for batch_count, real_batch in enumerate(batches):
            # Create a batch to feed the discriminator model
            if batch_count % 2 == 0:
                next_batch, y_disc, y_cat, y_cont = get_fake_disc_batch(
                    generator_model=self.generator.model,
                    batch_size=batch_size,
                    cat_dim=self.generator.cat_dim,
                    cont_dim=self.generator.cont_dim,
                    noise_dim=self.generator.noise_dim,
                    noise_scale=noise_scale)
            else:
                next_batch, y_disc, y_cat, y_cont = get_real_disc_batch(
                    real_batch=real_batch,
                    batch_size=batch_size,
                    cat_dim=self.generator.cat_dim,
                    cont_dim=self.generator.cont_dim,
                    noise_scale=noise_scale)

            # fake_batch, fake_y_disc, fake_y_cat, fake_y_cont = \
            #     get_fake_disc_batch(generator_model=self.generator.model,
            #                         batch_size=batch_size // 2,
            #                         cat_dim=self.generator.cat_dim,
            #                         cont_dim=self.generator.cont_dim,
            #                         noise_dim=self.generator.noise_dim,
            #                         noise_scale=noise_scale)
            #
            # real_batch, real_y_disc, real_y_cat, real_y_cont = \
            #     get_real_disc_batch(real_batch=real_batch,
            #                         batch_size=batch_size // 2,
            #                         cat_dim=self.generator.cat_dim,
            #                         cont_dim=self.generator.cont_dim,
            #                         noise_scale=noise_scale)
            # Combine the fake and real batches
            # next_batch = np.concatenate((fake_batch, real_batch))
            # y_disc = np.concatenate((fake_y_disc, real_y_disc))
            # y_cat = np.concatenate((fake_y_cat, real_y_cat))
            # y_cont = np.concatenate((fake_y_cont, real_y_cont))

            # Train the discriminator on real and fake images
            disc_loss = self.discriminator.model.train_on_batch(
                next_batch, [y_disc, y_cat, y_cont])

            # Create a batch to feed the generator model
            x_gen, y_gen, y_cat, y_cont, y_cont_target = get_gen_batch(
                batch_size=batch_size,
                cat_dim=self.generator.cat_dim,
                cont_dim=self.generator.cont_dim,
                noise_dim=self.generator.noise_dim,
                noise_scale=noise_scale)

            # Freeze the discriminator
            self.discriminator.trainable = False
            gen_loss = gan.train_on_batch([y_cat, y_cont, x_gen],
                                          [y_gen, y_cat, y_cont_target])

            # Unfreeze the discriminator
            self.discriminator.trainable = True

            progbar.add(batch_size, values=[("D tot", disc_loss[0]),
                                            ("D log", disc_loss[1]),
                                            ("D cat", disc_loss[2]),
                                            ("D cont", disc_loss[3]),
                                            ("G tot", gen_loss[0]),
                                            ("G log", gen_loss[1]),
                                            ("G cat", gen_loss[2]),
                                            ("G cont", gen_loss[3])])

    def _save_progress(self):
        """Save the progress of the generator and discriminator"""
        print("Saving progress to", str(self.save_dir))
        self.generator.model.save(self.save_dir / "generator.h5",
                                  include_optimizer=False)
        self.discriminator.model.save(self.save_dir / "discriminator.h5",
                                      include_optimizer=False)

    def fit(self, x_train, num_epochs=1, batch_size=128, noise_scale=0.5):
        """
        Method to train GAN.
        """

        # Set up the losses for models
        gan = self._setup_models()

        for epoch_n in range(num_epochs):
            print(f"Epoch {epoch_n + 1}")
            try:
                self._train_epoch(x_train, gan, batch_size, noise_scale)

                # Save every so often
                if epoch_n % 15 == 0:
                    self._save_progress()

                # Save screenshots
                if epoch_n % 1 == 0:
                    # Produce an output
                    fake_img = self.generator.model.predict(
                        [sample_cat(1, self.generator.cat_dim),
                         sample_noise(noise_scale, 1,
                                      self.generator.cont_dim),
                         sample_noise(noise_scale, 1,
                                      self.generator.noise_dim)],
                        batch_size=batch_size)[0]
                    fake_img += 1
                    fake_img /= 2
                    fake_img *= 255
                    fake_img = fake_img.astype(np.uint8)
                    cv2.imwrite(str(self.save_dir / (str(epoch_n) + ".png")),
                                fake_img)
            except KeyboardInterrupt:
                self._save_progress()
                exit(0)


def gaussian_loss(y_true, y_pred):
    Q_C_mean = y_pred[:, 0, :]
    Q_C_logstd = y_pred[:, 1, :]

    y_true = y_true[:, 0, :]

    epsilon = (y_true - Q_C_mean) / (K.exp(Q_C_logstd) + K.epsilon())
    loss_Q_C = (Q_C_logstd + 0.5 * K.square(epsilon))
    loss_Q_C = K.mean(loss_Q_C)

    return loss_Q_C


def sample_noise(noise_scale, batch_size, noise_dim):
    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim))


def sample_cat(batch_size, cat_dim):
    y = np.zeros((batch_size, cat_dim), dtype="float32")
    random_y = np.random.randint(0, cat_dim, size=batch_size)
    y[np.arange(batch_size), random_y] = 1
    return y


def get_gen_batch(batch_size, cat_dim, cont_dim, noise_dim, noise_scale):
    x_gen = sample_noise(noise_scale, batch_size, noise_dim)
    y_gen = np.zeros((x_gen.shape[0], 2), dtype=np.uint8)
    y_gen[:, 1] = 1

    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)

    # Repeat y_cont to accomodate for keras" loss function conventions
    y_cont_target = np.expand_dims(y_cont, 1)
    y_cont_target = np.repeat(y_cont_target, 2, axis=1)

    return x_gen, y_gen, y_cat, y_cont, y_cont_target


def get_real_disc_batch(real_batch, batch_size, noise_scale, cat_dim, cont_dim):
    y_disc = np.zeros((real_batch.shape[0], 2), dtype=np.uint8)
    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)

    # Produce an output
    y_disc[:, 1] = 1

    # Repeat y_cont to accomodate for keras" loss function conventions
    y_cont = np.expand_dims(y_cont, 1)
    y_cont = np.repeat(y_cont, 2, axis=1)

    return real_batch, y_disc, y_cat, y_cont


def get_fake_disc_batch(generator_model, batch_size,
                        cat_dim, cont_dim, noise_dim, noise_scale):
    # Pass noise to the generator
    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)
    noise_input = sample_noise(noise_scale, batch_size, noise_dim)

    # Produce an output
    fake_batch = generator_model.predict([y_cat, y_cont, noise_input],
                                         batch_size=batch_size)
    y_disc = np.zeros((fake_batch.shape[0], 2), dtype=np.uint8)
    y_disc[:, 0] = 1

    # Repeat y_cont to accomodate for keras" loss function conventions
    y_cont = np.expand_dims(y_cont, 1)
    y_cont = np.repeat(y_cont, 2, axis=1)

    return fake_batch, y_disc, y_cat, y_cont
