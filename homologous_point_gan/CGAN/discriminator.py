from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils.vis_utils import plot_model


def get_discriminator(generator_input_shape=(512, 512, 3), generator_output_shape=(512, 512, 1)):
	in_src_image = Input(shape=generator_input_shape)
	in_target_image = Input(shape=generator_output_shape)
	merged = Concatenate()([in_src_image, in_target_image])

	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	# Single output
	d = Flatten()(d)
	out = Dense(1, activation='sigmoid')(d)
	model = Model([in_src_image, in_target_image], out)
	opt = Adam(lr=0.00002)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model