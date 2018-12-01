import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class Kohonen_Map(object):

	map_units = 0

	# Função de criação e inicialização da classe
	def __init__(self,n_entries,input_dimension,size='small',epochs=2):
		
		self.n_entries = n_entries
		self.input_dimension = input_dimension
		print("n_entries=",n_entries)
		print("input_dimension=",input_dimension)

		self.heuristic_map_units(size)
		print("map_units=",self.map_units)

		self.map = np.random.uniform(size=(self.map_units,self.map_units,self.input_dimension))

		self.max_epochs = 100
		self.learning_rate = 0.1
		self.initial_radius = self.map_units/2


	def heuristic_map_units(self,size='small'):

		print("called")
		self.map_units = 5*self.n_entries**0.54321

		if size == 'big':
			self.map_units = self.map_units
		else:
			self.map_units = 0.25*self.map_units

		self.map_units = int(self.map_units)
		return

	def train(self, X):

		epoch = 0
		radius = self.initial_radius
		# Cada epoca representa uma passagem inteira pelo dataset
		while epoch < self.max_epochs:
			print("Epoch=",epoch)
			# Aleatoriza a passagem pelo dataset
			shuffle = np.random.randint(self.n_entries,size=self.n_entries)
			
			# Percorre o dataset inteiro
			for i in range(self.n_entries):
				if i%25 == 0 : print("\tIter=",i)
				# Verificar se o mapa convergiu

				# TO DO

				x_i = X[shuffle[i]]
				x_i_tile = np.tile(x_i,(self.map_units,self.map_units,1))
				D = np.linalg.norm(x_i_tile-self.map, axis=2)
				#print("D=",D)

				BMU = np.unravel_index(np.argmin(D,axis=None),D.shape)
				#print("BMU=",BMU)

				for i in range(self.map_units):
					for j in range(self.map_units):
						distance = np.linalg.norm([i-BMU[0],j-BMU[1]])
						if distance <= radius:
							self.map[i][j] = self.map[i][j] + self.learning_rate*(x_i-self.map[i][j])

				self.learning_rate = self.learning_rate*0.5
				radius = self.initial_radius*math.exp(-epoch/self.max_epochs)

			# Final do for que percorre o dataset inteiro por epoca
			epoch = epoch + 1



		return

	def get_painted_map(self, X, Y):

		BMU = np.zeros([2],dtype=np.int32)
		result_map = np.zeros([self.map_units,self.map_units,3],dtype=np.float32)

		for i in range(X.shape[0]):
			x_i = X[i]
			x_i_tile = np.tile(x_i,(self.map_units,self.map_units,1))
			D = np.linalg.norm(x_i_tile-self.map,axis=2)

			BMU = np.unravel_index(np.argmin(D,axis=None),D.shape)

			x = BMU[0]
			y = BMU[1]

			if Y[i] == 1:
				if result_map[x][y][0] <= 0.5:
					result_map[x][y] += np.asarray([0.5,0,0])
			elif Y[i] == 2:
				if result_map[x][y][1] <= 0.5:
					result_map[x][y] += np.asarray([0,0.5,0])
			elif Y[i] == 3:
				if result_map[x][y][2] <= 0.5:
					result_map[x][y] += np.asarray([0,0,0.5])


		result_map = np.flip(result_map,0)
		print("result_map",result_map)
		plt.imshow(result_map, interpolation='nearest')
		plt.show()

		return

	def test(self,X,Y):
		test_map = np.zeros([self.map_units,self.map_units,3],dtype=np.float32)
		for i, j in np.ndindex((self.map_units,self.map_units)):
			D = np.linalg.norm(X - self.map[i,j], axis = 1)
			J = D.argmin()

			if Y[J] == 1:
					test_map[i][j] += np.asarray([1,0,0])
			elif Y[J] == 2:
					test_map[i][j] += np.asarray([0,1,0])
			elif Y[J] == 3:
					test_map[i][j] += np.asarray([0,0,1])

		test_map = np.flip(test_map,0)
		print("test_map",test_map)
		plt.imshow(test_map, interpolation='nearest')
		plt.show()		


		return 


# Função para normalizar uma matriz por colunas
def normalize(X):
	for i in range(X.shape[1]):
		X[:,i] = (X[:,i] - np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))

	return X


##
#	MAIN PART
##


# Le o banco de dados
dataset = pd.read_csv('wine.data', names=['class','alcohol','malic acid','ash','alcalinity of ash','magnesium','total phenols','flavanoids','nonflavanoid phenols','proanthocyanins','color intensity','hue','od280/od315 of diluted wines','proline'])

Y = dataset['class']
X = dataset.drop(['class'],axis=1)
X = X.values
X = normalize(X)


kohonen = Kohonen_Map(n_entries=X.shape[0],input_dimension=X.shape[1])
kohonen.train(X)
print("Finished Training")
#kohonen.get_painted_map(X,Y)
kohonen.test(X,Y)
