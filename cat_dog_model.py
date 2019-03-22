import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2


CATEGORIES =["Dog","Cat"]


def view_training_data():
	for category in CATEGORIES:
		path = os.path.join(os.getcwd(),"PetImages/"+category)
		for img in os.listdir(path):
			# removing the color by adding the grascale
			img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
			plt.imshow(img_array,cmap="gray")
			plt.show()
			image_size = 50

			new_imgarray = cv2.resize(img_array,(image_size,image_size))
			plt.imshow(new_imgarray,cmap="gray")
			plt.show()
			# print(img_array)
			break
		break


training_data = []
image_size = 50


def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(os.getcwd(),"PetImages/"+category)
		class_name = CATEGORIES.index(category)
		for img in os.listdir(path):
			# removing the color by adding the grascale
			try:
				img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
				# plt.imshow(img_array,cmap="gray")
				# plt.show()

				new_img_array = cv2.resize(img_array,(image_size,image_size))

				training_data.append([new_img_array,class_name])
				# plt.imshow(new_imgarray,cmap="gray")
				# plt.show()
				# print(img_array)
			except Exception as e:
				pass

	print(len(training_data))


	import random

	random.shuffle(training_data)

	for sample in training_data:
		print(sample[1])


	x_train=[]
	y_train=[]

	# separate the features and labels from the trainind data
	for features,label in training_data:
		x_train.append(features)
		y_train.append(label)

				
	# converte the features into a numpy array
	# reshape(number of features (-1) means any number, size,size,(1 for grey scale 3 for color images))
	x_train = np.array(x_train).reshape(-1,image_size,image_size,1)
		

	import pickle


	pickle_save = open("cat_dog_features.pickle","wb")
	pickle.dump(x_train,pickle_save)
	pickle_save.close()

	pickle_save = open("cat_dog_labels.pickle","wb")
	pickle.dump(y_train,pickle_save)
	pickle_save.close()


	pickle_read = open("cat_dog_features.pickle","rb")
	x_train = pickle.load(pickle_read)

	print(x_train[1])


# create_training_data()


import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

# NAME ="Cats-vs-Dogs-cnn-64x2-{}".format(int(time.time()))

NAME ="PCC-vs-NCC-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


x_train = pickle.load(open("cat_dog_features.pickle","rb"))
y_train = pickle.load(open("cat_dog_labels.pickle","rb"))

# normalize the data , i.e putting it in range of o to 1, since this is imagery data , pixel range from 0 to 255, thus we can just divide the data set by 255 to get the normalized data

x_train = x_train/255.0


model = Sequential()

# layer 1
# input layer
model.add(Conv2D(64,(3,3),input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer2
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer3
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

# output layer
model.add(Dense(1))
model.add(Activation("sigmoid"))


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=10,batch_size=40,validation_split=0.1,callbacks=[tensorboard])


# model.save("64x3-CNN-MODEL")
model.save("64x3-CNN-MODEL-CC")




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
