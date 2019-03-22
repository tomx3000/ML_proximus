
import numpy as np 
import matplotlib.pyplot as plt 
# import matplotlib.animatiom as animation

# DDOS 
# how many packets are sent per second, size of packets


# using the elbow method to determine the best valuefor k
def best_k_value(k_max):
	# using the elbow method
	sets_k = {}
	for k in range(1,k_max):
		sets_k[k]=0
		clusters = kmeans(dataset,k)
		for cluster in clusters:
			# find the mean value i.e the error value i.e the distance btn the centroid and all its point
			mean = clusterMean(cluster)

			for datapoint in cluster:
				sets_k[k]+=(datapoint-mean)**2


	# the graph this sets of k , pick the k value at the elbow point


def load_data(txt_name):
	return np.loadtxt(txt_name)

def euclidian(a,b):
	return np.linalg.norm(a-b)


def kmeans(k,epslon=0,distance="euclidian"):
	# epslon == error for when to stop training
	history_centroids =[]

	distance_method =euclidian

	# loading the cyber security data , having the columns for number of packets , size of packets a
	dataset = load_data("cyber_data.txt")

	# getting the number of samples and the number of features within the data
	number_instances,number_features = dataset.shape

	# initializing the initial center points to  random tensors  
	current_centroids = dataset[np.random.randint(0,number_instances-1,size=k)]

	# adding centroids to a list that will keep track of them as they change in each epoch(iteration)
	history_centroids.append(current_centroids)

	previous_centroids = np.zeros(current_centroids.shape)


	clusters= np.zeros((number_instances,1))

	norm = distance_method(current_centroids,previous_centroids)

	iterations =0

	while norm > epslon:

		iterations+=1

		# norm = distance_method(current_centroids,previous_centroids)

		for data_index,data in enumerate(dataset):

			distance_vector = np.zeros((k,1))

			for centroid_index,centroid in enumerate(current_centroids):

				distance_vector[centroid_index] = distance_method(centroid,data)

			clusters[data_index,0] = np.argmin(distance_vector)
		
		# creating a tempory cluster for 

		temp_centroids = np.zeros((k,number_features)) 

		for c_index in range(len(current_centroids)):

			data_points_indices_close_together = [index for index in range(len(clusters)) if clusters[index] ==  c_index]


			current_centroid = np.mean(dataset[data_points_indices_close_together],axis=0)

			temp_centroids[c_index, :] =current_centroid

		current_centroids = temp_centroids

		history_centroids.append(temp_centroids)


		return current_centroids,history_centroids,clusters



def plot(dataset,history_centroids,clusters):
	color =['r','g']

	fig,ax = plt.subplots()

	for d_index in range(dataset.shape[0]):

		index_close_data = [index for index in range(len(clusters)) if clusters[index] == d_index]

		for index_index in index_close_data :
			ax.plot(dataset[index_index][0],dataset[index_index][1],(color[d_index]+'o'))

	plt.show()			


def execute():
	dataset = load_data('cyber_data.txt')
	centroids, history_centroids, belongs_to = kmeans(2)
	plot(dataset, history_centroids, belongs_to)



execute()
