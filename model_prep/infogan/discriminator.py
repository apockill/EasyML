import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dense, Activation, Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Deconv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU


class Discriminator:
    MODEL_NAME = "DCGAN_discriminator"

    def __init__(self, discriminator_model: Model):
        self.model = discriminator_model

    @classmethod
    def from_scratch(cls, cat_dim, cont_dim, img_size, img_ch, use_mbd=False):
        """
        Discriminator model of the DCGAN

        :param cat_dim: Latent categorical dimension
        :param cont_dim: Latent continuous dimension
        :param img_size: side length of the image
        :param use_mbd: Use mini batch disc
        :return: model (keras NN) the Neural Net model
        """

        disc_input = Input(shape=(img_size, img_size, img_ch),
                           name="discriminator_input")

        # Mnist
        # filters = [128]
        # stride = (2,2)
        # For other datasets
        filters = [64, 128, 256, 128]
        strides = [2, 2, 1, 1]

        # First conv
        x = Conv2D(64, (3, 3), strides=(2, 2), name="disc_Conv2D_1",
                   padding="same")(disc_input)
        x = LeakyReLU(0.2)(x)

        # Next convs
        for i, f in enumerate(filters):
            name = "disc_Conv2D_%s" % (i + 2)
            x = Conv2D(f, (3, 3),
                       strides=(strides[i], strides[i]),
                       name=name, padding="same")(x)
            x = BatchNormalization(axis=-1)(x)
            x = LeakyReLU(0.2)(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        def linmax(x):
            return K.maximum(x, -16)

        def linmax_shape(input_shape):
            return input_shape

        # More processing for auxiliary Q
        x_Q = Dense(128)(x)
        x_Q = BatchNormalization()(x_Q)
        x_Q = LeakyReLU(0.2)(x_Q)
        x_Q_Y = Dense(cat_dim, activation='softmax', name="Q_cat_out")(x_Q)
        x_Q_C_mean = Dense(cont_dim, activation='linear',
                           name="dense_Q_cont_mean")(x_Q)
        x_Q_C_logstd = Dense(cont_dim, name="dense_Q_cont_logstd")(x_Q)
        x_Q_C_logstd = Lambda(linmax, output_shape=linmax_shape)(x_Q_C_logstd)

        # Reshape Q to nbatch, 1, cont_dim[0]
        x_Q_C_mean = Reshape((1, cont_dim))(x_Q_C_mean)
        x_Q_C_logstd = Reshape((1, cont_dim))(x_Q_C_logstd)
        x_Q_C = concatenate([x_Q_C_mean, x_Q_C_logstd], name="Q_cont_out",
                            axis=1)

        def minb_disc(z):
            diffs = K.expand_dims(z, 3) - K.expand_dims(
                K.permute_dimensions(z, [1, 2, 0]), 0)
            abs_diffs = K.sum(K.abs(diffs), 2)
            z = K.sum(K.exp(-abs_diffs), 2)

            return z

        def lambda_output(input_shape):
            return input_shape[:2]

        num_kernels = 300
        dim_per_kernel = 5

        M = Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)
        MBD = Lambda(minb_disc, output_shape=lambda_output)

        if use_mbd:
            x_mbd = M(x)
            x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
            x_mbd = MBD(x_mbd)
            x = concatenate([x, x_mbd])

        # Create discriminator model
        x_disc = Dense(2, activation='softmax', name="disc_out")(x)
        discriminator_model = Model(inputs=[disc_input],
                                    outputs=[x_disc, x_Q_Y, x_Q_C],
                                    name=cls.MODEL_NAME)

        return cls(discriminator_model=discriminator_model)

    @classmethod
    def from_path(cls, model_path):
        model = load_model(model_path)
        return cls(model)
