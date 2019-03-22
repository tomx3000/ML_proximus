
import numpy as np
import pandas as pd 
import warnings
from patsy import matrices

def sigmoid(x):
	# sigmoid function of x
	return 1/(1+np.exp(-x))


def dataGeneration():
	# allows for repeated random number generation , which is good during testing since the same data will be generated each time
	np.random.seed(o)
	# convergence tolerenace
	# minimum threshhold btn predicted output and actual output...this will tel our model when to stop learning
	convergence_tolerance=1e-8

	l2_regularization = None

	training_iterations = 20



	# data definintion varibles
	# measure of how two varibles move togeteher
	# + cov (move in same direction)
	# - cov (move in opposite directions)
	covariance = 0.95

	size_data_generated = 1000

	variance = 1 (variance of the noise , how spread out the data is_)


	## model settings
	# assuming this values from the ligresion lline
	# i.e b0 + b1x + b3x2 +......bnxn
	beta_x, beta_z, beta_v = -4, .9, 1 # true beta coefficients
	var_x, var_z, var_v = 1, 1, 4 # variances of inputs



	## the model specification you want to fit
	# x = height, z = weight , v = blood pressure
	formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'

	# a distribution is a funciotn that provides us the probabilities of all possible outcomes of a stochastic process(cannot be predicted.)

	# deterministic process (can be predicted)

	# 
