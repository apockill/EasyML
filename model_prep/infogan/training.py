from pathlib import Path
from time import time

import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import generic_utils
from keras.callbacks import TensorBoard

from easy_ml.train_utils.batching import batch_iterator
from infogan.batch_utils import sample_noise, sample_cat, get_gen_batch, \
    get_real_disc_batch, get_fake_disc_batch

from model_prep.infogan.generator import GeneratorDeconv
from model_prep.infogan.discriminator import Discriminator


class Trainer:
    def __init__(self, generator, discriminator, save_dir, batch_size=128,
                 noise_scale=.5, num_epochs=128, label_flipping=0,
                 label_smoothing=False, summary_freq=15):
        """
        :param num_epochs: Total number of epochs to train for
        :param label_flipping: If True, sometimes the discriminator will be fed
        the incorrect information, to make it harder to train.
        :param label_smoothing: If True, labels will be between 0.9 -> 1.0
        :param summary_freq: Number of batches between writing a summary
        """

        # Modeols
        self.generator: GeneratorDeconv = generator
        self.discriminator: Discriminator = discriminator

        # Recording information
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard = TensorBoard(log_dir=str(self.save_dir))
        self.summary_freq = summary_freq

        # Training Parameters
        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.num_epochs = num_epochs
        self.label_flipping = label_flipping
        self.label_smoothing = label_smoothing

        # Current State of Training
        self.curr_epoch = 0  # Total epochs done so far
        self.curr_batch = 0  # Total batches done so far

    def _setup_models(self):
        """
        Set up models for training
        """

        # Create the DCGAN model
        gan = Model(inputs=self.generator.model.inputs,
                    outputs=self.discriminator.model(
                        self.generator.model(self.generator.model.inputs)),
                    name="DCGAN")
        self.tensorboard.set_model(gan)

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

    def _train_epoch(self, x_train, gan):
        # Create a progress bar for the Epoch
        progbar = generic_utils.Progbar(x_train.shape[0])

        # Create a batch iterator
        batches = batch_iterator(x_train, self.batch_size)

        for batch_count, real_batch in enumerate(batches):
            # Create a batch to feed the discriminator model
            if batch_count % 2 == 0:
                next_batch, y_disc, y_cat, y_cont = get_fake_disc_batch(
                    generator_model=self.generator.model,
                    batch_size=self.batch_size,
                    cat_dim=self.generator.cat_dim,
                    cont_dim=self.generator.cont_dim,
                    noise_dim=self.generator.noise_dim,
                    noise_scale=self.noise_scale,
                    label_flipping=self.label_flipping)
            else:
                next_batch, y_disc, y_cat, y_cont = get_real_disc_batch(
                    real_batch=real_batch,
                    batch_size=self.batch_size,
                    cat_dim=self.generator.cat_dim,
                    cont_dim=self.generator.cont_dim,
                    noise_scale=self.noise_scale,
                    label_flipping=self.label_flipping,
                    label_smoothing=self.label_smoothing)

            # Train the discriminator on real and fake images
            disc_loss = self.discriminator.model.train_on_batch(
                next_batch, [y_disc, y_cat, y_cont])

            # Create a batch to feed the generator model
            x_gen, y_gen, y_cat, y_cont, y_cont_target = get_gen_batch(
                batch_size=self.batch_size,
                cat_dim=self.generator.cat_dim,
                cont_dim=self.generator.cont_dim,
                noise_dim=self.generator.noise_dim,
                noise_scale=self.noise_scale)

            # Freeze the discriminator
            self.discriminator.trainable = False
            gen_loss = gan.train_on_batch([y_cat, y_cont, x_gen],
                                          [y_gen, y_cat, y_cont_target])

            # Unfreeze the discriminator
            self.discriminator.trainable = True

            # Log
            logs = [("D tot", disc_loss[0]),
                    ("D log", disc_loss[1]),
                    ("D cat", disc_loss[2]),
                    ("D cont", disc_loss[3]),
                    ("G tot", gen_loss[0]),
                    ("G log", gen_loss[1]),
                    ("G cat", gen_loss[2]),
                    ("G cont", gen_loss[3])]
            progbar.add(self.batch_size, values=logs)
            if self.curr_batch % self.summary_freq == 0:
                self._write_summary(*zip(*logs), self.curr_batch)
            self.curr_batch += 1

    def save_progress(self, gen_name="generator", disc_name="discriminator"):
        """Save the progress of the generator and discriminator"""
        print("\nSaving progress to", str(self.save_dir))
        self.generator.model.save(self.save_dir / (gen_name + ".h5"),
                                  include_optimizer=False)
        self.discriminator.model.save(self.save_dir / (disc_name + ".h5"),
                                      include_optimizer=False)

    def _write_summary(self, names, logs, batch_no):
        """
        :param names: example ["a_loss", "b_loss", "c_loss"]
        :param logs: [#, #, #]
        :param batch_no: #
        """
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.tensorboard.writer.add_summary(summary, batch_no)
            self.tensorboard.writer.flush()

    def fit(self, x_train):
        """
        Method to train GAN.
        """

        # Set up the losses for models
        gan = self._setup_models()

        for self.curr_epoch in range(self.num_epochs):
            print(f"Epoch {self.curr_epoch + 1}")
            try:
                self._train_epoch(x_train, gan)

                # Save every so often
                if self.curr_epoch % 15 == 0:
                    self.save_progress()

                # Save an image every epoch
                fake_img = self.generator.model.predict(
                    [sample_cat(1, self.generator.cat_dim),
                     sample_noise(self.noise_scale, 1,
                                  self.generator.cont_dim),
                     sample_noise(self.noise_scale, 1,
                                  self.generator.noise_dim)],
                    batch_size=self.batch_size)[0]
                fake_img = (fake_img + 1) * 127.5
                fake_img = fake_img.astype(np.uint8)
                img_path = self.save_dir / (str(int(time())) + ".png")
                cv2.imwrite(str(img_path), fake_img)

            except KeyboardInterrupt:
                self.save_progress()
                exit(0)



def gaussian_loss(y_true, y_pred):
    Q_C_mean = y_pred[:, 0, :]
    Q_C_logstd = y_pred[:, 1, :]

    y_true = y_true[:, 0, :]

    epsilon = (y_true - Q_C_mean) / (K.exp(Q_C_logstd) + K.epsilon())
    loss_Q_C = (Q_C_logstd + 0.5 * K.square(epsilon))
    loss_Q_C = K.mean(loss_Q_C)

    return loss_Q_C
