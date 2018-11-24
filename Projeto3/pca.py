import numpy as np 
import numpy.linalg as la ##para autovetor e autovalor 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler ### para normalizar os dados
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##
#	Adaptative PCA PART
##
class APCA(object):

	def __init__(self,input_lenght=13,output_lenght=8,max_iters=500):

		# Inicializando valores da arquitetura
		self.input_lenght = input_lenght
		self.output_lenght = output_lenght
		self.max_iters = max_iters

		# Matrizes de pesos e pesos laterais
		self.weights = np.random.rand(self.input_lenght,self.output_lenght)
		self.u = np.random.rand(self.output_lenght,self.output_lenght)

		# Deixando a matriz de pesos laterais triangular
		for i in range(0,self.output_lenght):
			for j in range(i+1,self.output_lenght):
				self.u[i][j] = 0


		# Normalizando os valores dos pesos pelas colunas
		for i in range(self.weights.shape[1]):
			self.weights[:,i] = (self.weights[:,i] - np.amin(self.weights[:,i])) / (np.amax(self.weights[:,i]) - np.amin(self.weights[:,i]))


		# Inicializando parametros
		self.eta = 0.0001
		self.mi = 0.0002
		self.beta = 0.0001


		return

	def forward(self,x):
		y = np.dot(x,self.weights)
		#print("y=",y)

		for i in range(0,y.size):
			total_u = 0
			#print("y[",i,"] = y",end="")
			for j in range(0,i+1):
				#print(" + u[",i,"][",j,"] * y[",j,"]",end="")
				total_u = total_u + self.u[i][j]*y[j]

			#print("")
			y[i] = y[i] + total_u

		return y

	def train(self,data,alpha=0.001):
		
		X = data.copy()
		del X['class']
		#print("X = ",X)
		X = X.values

		#Normalizando o banco de dados
		for i in range(X.shape[1]):
			X[:,i] = (X[:,i] - np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))

		dW = np.zeros((self.input_lenght,self.output_lenght))
		dU = np.zeros((self.output_lenght,self.output_lenght))
		counter = 0
		while counter < self.max_iters:
			
			# Pega uma entrada do dataset aleatoria para testar
			ids = np.random.randint(X.shape[0])
			x_i = X[ids]

			# Calcula a saida daquela entrada
			y = self.forward(x_i)
			#print("y=",y)

			# Calculando o delta dos pesos das camadas
			for i in range(0,self.input_lenght):
				for j in range(0,self.output_lenght):
					dW[i][j] = self.eta*x_i[i]*y[j] + self.beta*dW[i][j]

			#print("dW=",dW)
			#Atualizando os pesos
			self.weights += dW

			# Normalizando os novos pesos
			for i in range(self.weights.shape[1]):
				self.weights[:,i] = (self.weights[:,i] - np.amin(self.weights[:,i])) / (np.amax(self.weights[:,i]) - np.amin(self.weights[:,i]))
			#print("W=",self.weights)
			# Atualizando os pesos laterais
			err = 0
			for l in range(0,self.output_lenght):
				for j in range(0,l+1):
					dU[l][j] = -(self.mi*y[l]*y[i]) + self.beta*dU[l][j]

			self.u += dU
			err = sum(sum(self.u))
			#print("err=",err)

			#print("dU=",dU)

			# Atualizando parametros finais
			self.eta = max(self.eta*alpha, 0.0001)
			self.mi = max(self.mi*alpha, 0.0002)
			self.beta = max(self.beta*alpha, 0.0001)

			counter = counter+1

		return

	def run(self,data):

		X = data.copy()
		del X['class']
		X = X.values


		#Normalizando o banco de dados
		for i in range(X.shape[1]):
			X[:,i] = (X[:,i] - np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))

		new_data = []

		for i in range(0,X.shape[0]):
			y = self.forward(X[i])
			#print("Y=",y)
			#new_data = np.append(new_data,y,axis=0)
			new_data.append(y)

		new_data = np.array(new_data)
		return new_data


##
#	Classic PCA PART
##

def classic_pca(data):


	scaler = StandardScaler()
	#df = pd.read_csv('wine.data', names=['class','alcohol','malic acid','ash','alcalinity of ash','magnesium','total phenols','flavanoids','nonflavanoid phenols','proanthocyanins','color intensity','hue','od280/od315 of diluted wines','proline'])
	df = data.copy()

	#print('wine original:','\n' ,df) 

	#print(df.loc[0,"class"]) # Prints Dataset Sizes
	del df['class'] ##apagando a coluna de strings

	scaler.fit(df)
	wine = scaler.transform(df) #####Normaliza os dados
	#print("dados transformado:",'\n',wine,'\n')

	mat_cov = np.cov(np.transpose(wine)) ###### Matriz de covariância
	#print("matriz de covariância:","\n",mat_cov, "\n")

	val,vet=la.eig(mat_cov)
	#print("Autovalores: ","\n", val)
	#print("Autovetores:","\n", vet)


	por = val/sum(val)

	n_components = 0
	total_por = 0
	while total_por < 0.9:
		total_por = total_por + por[n_components]
		n_components = n_components + 1


	#print("porcentagem=",por)
	#print("sum=",total_por)
	#print("n_components=",n_components)

	vet = vet[:,0:n_components]
	#print("new vet=",np.transpose(vet),"\n")

	new_data = np.transpose(np.matmul(np.transpose(vet),np.transpose(wine)))

	classes = np.zeros(wine.shape[0])

	for i in range(0,wine.shape[0]):
		if data['class'][i] == 1:
			classes[i] = 1
		elif data['class'][i] == 2:
			classes[i] = 2
		elif data['class'][i] == 3:
			classes[i] = 3


	new_data = np.insert(new_data,new_data.shape[1],classes,axis=1)

	return new_data


def normalize(X):
	for i in range(X.shape[1]):
		X[:,i] = (X[:,i] - np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))

	return X

##
#	MAIN PART
##

dataset = pd.read_csv('wine.data', names=['class','alcohol','malic acid','ash','alcalinity of ash','magnesium','total phenols','flavanoids','nonflavanoid phenols','proanthocyanins','color intensity','hue','od280/od315 of diluted wines','proline'])

pca_data = classic_pca(dataset)
apca = APCA()
apca.train(dataset)
apca_data = apca.run(dataset)
#print("classic_pca=",pca_data.shape)
#print("apca_data=",apca_data.shape)

###########################################

# APLICANDO MLP NO DATASET SEM PCA

Y = dataset['class']
X = dataset.drop(['class'],axis=1)
X = X.values
X = normalize(X)


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.25, random_state=27)

clf = MLPClassifier(max_iter=500, alpha=0.001, tol=0.000000001)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accurary without pca:",accuracy_score(y_test, y_pred))

###########################################

# APLICANDO MLP NO DATASET COM PCA ADAPTATIVA APLICADA

X = apca_data

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.25, random_state=27)

clf = MLPClassifier(max_iter=500, alpha=0.001, tol=0.000000001)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accurary with pca:",accuracy_score(y_test, y_pred))


###########################################