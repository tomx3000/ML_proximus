import numpy as np 
import matplotlib.pyplot as plt

def getData():
	points = np.genfromtxt("data.csv", delimiter=",")
	return points

def error_of_the_line_given_points(b,m,points):
	# eqn of line y = mx + b
	# get alllpoints
	# calculate the error using the mean square error equation
	error = 0
	for index in range(0,len(points)):
		x= points[index,0]
		y= points[index,1]
		# error = E(y-y*)^2
		error +=(y -(m*x + b))**2
	
	return error/floatlen(points)

def runmer():
	# get all the points
	# run them through the gradient decent algorithm 
	# on each iteration get the new line i.e b and m
	# termination is based on a set number of iteration
	# also add some learning rate
	number_of_iterations = 100 
	b_start = 0
	m_start = 0
	points = getData()
	error = 0

	for index in range(number_of_iterations):
		b_start,m_start,error = simple_gradient_descent(points,b_start,m_start)
		print("b: {} m: {} error: {} ".format(b_start,m_start,error))

	x= points[:,0]
	y= points[:,1]
	plt.scatter(x,y)
	# x.sort()
	# y.sort()
	x_values =[np.min(x),np.max(x)]
	y_values =[m_start*np.min(x)+b_start,m_start*np.max(x)+b_start]
	plt.plot(x_values,y_values,color="red")
	plt.show()

def get_best_fit_line(points,b_start,m_start):
	# apply an optimization algorithm for the pints to get appropriate b and m values
	return simple_gradient_descent(points,b_start,m_start)

def simple_gradient_descent(points,b_start,m_start,learning_rate=0.0001):
	# get all the points
	N =len(points)
	error = 0
	b_grad=0
	m_grad=0
	# get the running rate if available
	# calculate the partial derivative for each optimized variable .i.e b and m 
	for index in range(len(points)):
		x = points[index,0]
		y = points[index,1]
		
		b_grad += (-2/N) * (y -(m_start * x )+b_start)
		m_grad += (-2/N) * x *(y -(m_start * x )+b_start)

		error+= (y-(m_start*x)+b_start)**2

	# then descent down the valley 
	b_start = b_start- (learning_rate* b_grad)
	m_start = m_start - (learning_rate * m_grad)
	error = error/N

	return [b_start,m_start,error]


def get_sample_svm_data():

	x_feature_set =np.array([ [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1]])
	y_sample_output_set = np.array([-1,-1,1,1,1])

	for pos,feature in enumerate(x_feature_set):
		if pos < 2:
			plt.scatter(feature[0],feature[1],marker="_",s=120)
		else:
			plt.scatter(feature[0],feature[1],marker="+",s=120)


	# plt.plot([-2,6],[6,0.5])


	svm_model = support_vector_machine(x_feature_set,y_sample_output_set)


	# drawing the hyperplane of the learned model
	x2=[svm_model[0],svm_model[1],-svm_model[1],svm_model[0]]
	x3=[svm_model[0],svm_model[1],svm_model[1],-svm_model[0]]

	x2x3 =np.array([x2,x3])
	X,Y,U,V = zip(*x2x3)
	# gca means get current axes
	ax = plt.gca()
	ax.quiver(X,Y,U,V,scale=1, color='blue')

	plt.show()


def support_vector_machine(x_feature_set,y_classification_set):

	# initializing the weights
	weights = np.zeros(len(x_feature_set[0]))
	# initialize the learing rate
	l_rate = 1
	# inintialize the epochs
	epochs = 100000

	errors = []

	for epoch in range(1,epochs):
		error = 0
		for pos,feature in enumerate(x_feature_set):
			if (y_classification_set[pos] * np.dot(feature,weights)) < 1:
				# miss-clasification
				weights = weights +  l_rate * (y_classification_set[pos] * feature - 2 * (1/epoch) * weights) 
				error=1

			else:
				# correctly-classified
				weights = weights - (2* l_rate * (1/epoch) * weights )

		# print("epoch: {}  error: {}".format(epoch,error))

		errors.append(error)

	# plt.plot(errors, '|')
	# plt.ylim(0.5,1.5)
	# # plt.axes().set_yticklabels([])
	# plt.show()

	return weights  


def getDataInOrder():
	# get all data points
	pass


if __name__ == '__main__':
	# runmer()
	get_sample_svm_data()
