import tensorflow as tf
mnist = tf.keras.datasets.mnist



def train_predict():
	print("welcome")

	(x_train,y_train),(x_test,y_test) = mnist.load_data()

	try:
		new_model= tf.keras.models.load_model("mnist_dataset_model")
		prediction = new_model.predict([x_test])

		print(prediction)

	except:

		# normalize the data sets
		# this particular normalization will simply make sure that the values are between 0 and 1
		# not a must but it has been proven helpfull for a better learning algorithim

		x_train = tf.keras.utils.normalize(x_train,axis=1)
		x_test = tf.keras.utils.normalize(x_test,axis=1)


		# next we build the actual model
		# we going to build a feed forwad model aka sequential model

		model = tf.keras.models.Sequential()

		# first we flatted the inputs i.e the 2x2 dimension array is flattened into a single dimension stream of inputs 784 (28x28)inputs to be exact

		model.add(tf.keras.layers.Flatten())

		# first layer
		model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
		# second layer
		model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
		# output layer
		# using an activiation of softmax due to the fact that it gives as some kind of probability distriution along the output
		model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

		# optimizer of cause for otimizing the nneural network

		# metrics, here we provide a metric we want to track eg. accuracy
		model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


		model.fit(x_train,y_train,epochs=3)


		# calculating validation loss and accuracy to hep us see whether the model has overfitted or not
		# i.e just trying to see if the model has learned any thing or simpy it just memorized the data

		val_loss,val_acc = model.evaluate(x_train,y_train)
		print(val_loss,val_acc)

		predictions = model.predict([x_test])

		print(predictions)

		import	 numpy as np 

		print(np.argmax(predictions[0]))

		import matplotlib.pyplot as plt
		plt.imshow(x_test[0])
		plt.show()	

		# model.save_weights("mnist_dataset_model")





def show_data(x_train):	
	import matplotlib.pyplot as plt 
	# just printing the data set as anrray of values
	# print(x_train[0])
	# graphing the data set
	plt.imshow(x_train[0],cmap=plt.cm.binary)
	plt.show()


train_predict()