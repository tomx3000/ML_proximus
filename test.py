
def optimizing_the_cnn():
	import time

	dense_layers = [0,1,2]
	layer_sizes = [32,64,128]
	conv_layers =[1,2,3]

	for dense_layer in dense_layers:
		for layer_size in layer_sizes:
			for conv_layer in conv_layers:

				NAME = "{}:conv {}:nodes {}:dense {}:times".format(conv_layer,layer_size,dense_layer,int(time.time()))


				print(NAME)


def use_cats_dog_model():
	import cv2
	import tensorflow as tf 
	CATEGORIES =["Dog","Cat"]

	def prepare(filepath):
		image_size = 50
		img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
		new_array = cv2.resize(img_array,(image_size,image_size))

		return new_array.reshape(-1,image_size,image_size,1)


	model = tf.keras.models.load_model("64x3-CNN-MODEL")
	prediction = model.predict([prepare("dog.jpeg")])
	print(CATEGORIES[int(prediction[0][0])])	


use_cats_dog_model()