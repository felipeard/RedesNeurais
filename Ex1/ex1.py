# Exercio 1 de Redes Neurais
# Felipe Alegria Rollo Dias 	nusp 9293501

# Libraries needed 
#from random import *
import numpy as np
import os

# Perceptron activation function 
def fnet(net, threshold=0):
	if net >= threshold:
		return (1)
	return (-1)

# Reads the contents from a file and transforms in matrix
def matrix(contents):
	return [item.split() for item in contents.split('\n')[:-1]]

# Reads the last line from the file which contains the waited answer
def answer(contents):
	lines = contents.splitlines()
	return lines[-1]

# Function that evaluates the net value of the neuron
#	using a matrix of values and a matrix of weights
def net(X, w, size):
	net = 0
	for i in range(0,size):
		for j in range(0,size):
			net = net + X[i][j]*w[i][j]
	return net

# Function that does the training utilizing the Adeline method
def adeline_train(eta, max_error=0.001, max_iter=200):
	# Opening just the first example to know how many lines there is
	data = open("example1.in").read() 
	X = matrix(data)
	nX = len(X)
	# Initialize the weights with random values
	weights = np.random.uniform(-0.5,0.5,nX*nX)
	weights = np.reshape(weights,(nX,nX)) # reshape the weights vector so it becomes a matrix as the entry data
	
	# Initialize counter and variables needed
	counter = 0 # counter iterations
	total_error = 2*max_error # Total error in a iteration trough all entries

	# While the error is too much or didn't passed the max number of iterations
	while total_error > max_error and counter < max_iter:
		n_files = 0 # number of files
		# For each file cointaining an entry
		for file in os.listdir():
			# Only works if the file is a entry file (*.in)
			if(file.endswith('.in')):
				n_files = n_files+1; # counting the number of files/entries
				# Reading the file with the training data
				data = open(file).read()
				X = matrix(data) # Reading the data from the file
				X = np.array(X) # Transforming to a NP array type data
				X = X.astype(np.float)	# Transforming from string to float
				Y = answer(data) # Reads the waited answer
				Y = np.array(Y) # Transform to a NP array
				Y = Y.astype(np.float) # Casts to float

				# Finds the net value and them apply the activation function to find the obtained answer
				net_ = net(X,weights,nX)
				y = fnet(net_);

				# Evaluate the error from this training entry
				error = Y - y
				total_error = total_error + error*error
				#print('net=',net_,'\nY=',Y,'\ny=',y,'\n')

				# Training part
				# Using the Delta Rule to update the weights
				dE2_dweights = -2*error*X
				weights = weights-eta*dE2_dweights
		# After running all file/entries
		# Evaluates the total error and adds the counter
		total_error = total_error/n_files
		counter = counter+1
	# After the while return the obtained weight with the minimum error
	return weights

def adeline_test(weights):
	for file in os.listdir():
		if(file.endswith('.test')):
			# Reading the file with the training data
			data = open(file).read()
			X = matrix(data) # Reading the data from the file
			X = np.array(X) # Transforming to a NP array type data
			X = X.astype(np.float)	# Transforming from string to float
			Y = answer(data) # Reads the waited answer
			Y = np.array(Y) # Transform to a NP array
			Y = Y.astype(np.float) # Casts to float

			# Finds the net value and them apply the activation function to find the obtained answer
			net_ = net(X,weights,len(X))
			y = fnet(net_);

			# Prints the results
			print('Testing ',file,'\n')
			print('Expected answer: ', Y,'\n')
			print('Obtained answer: ', y,'\n\n')
	return

##############################################################################
# MAIN PART

np.random.seed() # Generate seed for randomness
weights = adeline_train(0.1)
adeline_test(weights)