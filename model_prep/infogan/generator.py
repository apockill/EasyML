import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, concatenate
from keras.optimizers import Adam, SGD
from keras.layers.core import Flatten, Dense, Activation, Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Deconv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU

from easy_ml.convolution_utils import calc_conv, calc_deconv


def get_num_convolutions(filter_dim, stride_dim, padding, min_in_dim, out_dim):
    """A function for estimating the number of deconvolutions are necessary
    to get FROM min_in_dim to out_dim.
    :returns count of convolutions, input_dim"""
    input_dim = out_dim
    count = 0
    while True:
        new_dim = calc_conv(filter_dim, stride_dim, input_dim, padding)
        if new_dim[0] < min_in_dim[0] or new_dim[1] < min_in_dim[1]:
            break
        input_dim = new_dim
        count += 1
    return count, input_dim


class GeneratorDeconv:
    MODEL_NAME = "generator_deconv"

    def __init__(self, generator_model: Model):
        self.model = generator_model

        # Useful parts of the model
        self.cat_dim = self.model.get_layer("cat_input").input_shape[1]
        self.cont_dim = self.model.get_layer("cont_input").input_shape[1]
        self.noise_dim = self.model.get_layer("noise_input").input_shape[1]

    @classmethod
    def from_scratch(cls, cat_dim, cont_dim, noise_dim, img_size, img_ch,
                     batch_size):
        """
        Generator model of the DCGAN

        :param cat_dim: Latent categorical dimension
        :param cont_dim: Latent continuous dimension
        :param noise_dim: Noise dimension
        :param img_size: Image width == height (only specify for CelebA)
        :param start_size: The side resolution at which the deconvolutions start
        :param batch_size: Batch size that the model can take
        :return: model (keras NN) the Neural Net model
        """

        # Set up modifiable parameters
        f = 128
        nb_upconv = 4
        start_size = 4
        filter_dim = (3, 3)
        stride_dim = (2, 2)

        # Create the network
        cat_input = Input(shape=(cat_dim,), name="cat_input")
        cont_input = Input(shape=(cont_dim,), name="cont_input")
        noise_input = Input(shape=(noise_dim,), name="noise_input")

        gen_input = concatenate([cat_input, cont_input, noise_input])

        x = Dense(1024)(gen_input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Dense(f * start_size * start_size)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Reshape((start_size, start_size, f))(x)

        # Transposed conv blocks
        for i in range(nb_upconv - 1):
            nb_filters = int(f / (2 ** (i + 1)))
            img_size = start_size * (2 ** (i + 1))
            o_shape = (batch_size, img_size, img_size, nb_filters)
            x = Deconv2D(nb_filters, filter_dim, output_shape=o_shape,
                         strides=stride_dim,
                         padding="same")(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)

        # Last block
        img_size = start_size * (2 ** (nb_upconv))
        o_shape = (batch_size, img_size, img_size, img_ch)
        x = Deconv2D(img_ch, (3, 3), output_shape=o_shape,
                     strides=(2, 2),
                     padding="same")(x)
        x = Activation("tanh")(x)

        generator_model = Model(inputs=[cat_input, cont_input, noise_input],
                                outputs=[x], name=cls.MODEL_NAME)

        return cls(generator_model=generator_model)

    @classmethod
    def from_path(cls, model_path):
        model = load_model(model_path)
        return cls(model)
