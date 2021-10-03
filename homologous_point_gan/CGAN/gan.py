from keras.optimizers import Adam
from keras.models import Model
from keras.models import Input
from keras.layers import BatchNormalization
from homologous_point_gan.CGAN.generator import get_generator
from homologous_point_gan.CGAN.discriminator import get_discriminator


def get_gan(g_model, d_model, input_shape=(512, 512, 3), output_shape=(512, 512, 1)):

    g_model = get_generator(input_shape)
    d_model = get_discriminator(input_shape, output_shape)

	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False

	input_image = Input(shape=input_shape)
	generator_out = g_model(input_image)
	discriminator_out = d_model([input_image, generator_out])
    
	# src image as input, generated image and classification output
	gan_model = Model(input_image, [discriminator_out, generator_out])
	opt = Adam(lr=0.0002, beta_1=0.5)
	gan_model.compile(loss=['binary_crossentropy'], optimizer=opt, loss_weights=[1,100])
	return g_mode, d_model, gan_model