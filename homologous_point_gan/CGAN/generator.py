from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Flatten


def _get_conv_unit(l, n_filters=64, size_filter=4, stride=2, batch_norm=False):

    if batch_norm:
        l = BatchNormalization()(l)

    l = Conv2D(n_filters,\
        (size_filter,size_filter),\
        strides=(stride,stride),\
        padding='same',\
        kernel_initializer=RandomNormal(stddev=0.02))(l)
    return Activation('relu')(l)


def _get_conv_transpose(n_filters=64, size_filter=4, stride=2):
    return Conv2DTranspose(n_filters,\
        (size_filter,size_filter),\
        strides=(stride,stride),\
        padding='same',\
        kernel_initializer=RandomNormal(stddev=0.02))


def get_encoder(input_shape=(512, 512, 3)):
    in_image = Input(shape=input_shape)
    l = _get_conv_unit(in_image, n_filters=64, size_filter=5, stride=2, batch_norm=False)
    l = _get_conv_unit(l, n_filters=128, size_filter=5, stride=2, batch_norm=False)
    l = _get_conv_unit(l, n_filters=256, size_filter=5, stride=2, batch_norm=False)
    l = _get_conv_unit(l, n_filters=512, size_filter=5, stride=2, batch_norm=False)
    l = _get_conv_unit(l, n_filters=512, size_filter=5, stride=2, batch_norm=False)
    l = _get_conv_unit(l, n_filters=512, size_filter=5, stride=2, batch_norm=False)
    l = _get_conv_unit(l, n_filters=512, size_filter=5, stride=2, batch_norm=False)
    l = _get_conv_unit(l, n_filters=512, size_filter=5, stride=2, batch_norm=False)
    l = _get_conv_unit(l, n_filters=512, size_filter=5, stride=2, batch_norm=False) # 1 x 1 x 512
    return in_image, l


def get_decoder(encoded_in):
    l = _get_conv_transpose(n_filters=512, size_filter=5, stride=2)(encoded_in)
    l = _get_conv_transpose(n_filters=512, size_filter=5, stride=2)(l)
    l = _get_conv_transpose(n_filters=512, size_filter=5, stride=2)(l)
    l = _get_conv_transpose(n_filters=512, size_filter=5, stride=2)(l)
    l = _get_conv_transpose(n_filters=512, size_filter=5, stride=2)(l)
    l = _get_conv_transpose(n_filters=512, size_filter=5, stride=2)(l)
    l = _get_conv_transpose(n_filters=256, size_filter=5, stride=2)(l)
    l = _get_conv_transpose(n_filters=128, size_filter=5, stride=2)(l)
    l = _get_conv_transpose(n_filters=64, size_filter=5, stride=2)(l)
    l = _get_conv_transpose(n_filters=1, size_filter=5, stride=2)(l) # Output is 512 x 512 x 1
    l = Activation('sigmoid')(l)
    return l


def get_generator(input_shape=(512, 512, 3)):
    in_image, latent_out = get_encoder(input_shape)
    out_image = get_decoder(latent_out)
	model = Model(in_image, out_image)
    return model