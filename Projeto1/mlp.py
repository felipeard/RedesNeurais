# Exercicio 2 da matéria de Redes Neurais
#	Felipe Alegria Rollo Dias	nUSP 9293501
#	Leonardo Piccaro Rezende	nUSP 9364611
#

import numpy as np
import os
import random
from prettytable import PrettyTable
from sklearn.preprocessing import scale

# Classe que representa uma MLP com sua propria arquitetura
class MLP(object):

	# Dictionary do python para armazenar o modelo da MLP
	model = {}
	
	# Construtor da classe
	def __init__(self):
		# Iniatialize MLP class
		#self.architecture()
		return

	# Função de Ativação dos neuronios da MLP
	def fnet(net):
		return (1 / (1 + np.exp(-net)))

	# Função para calcula a derivada
	def df_dnet(f_net):
		return (f_net * (1 - f_net))

	# Função que inicializa a arquitetura da MLP baseado no problema especifico
	def architecture(self,input_lenght=13,hidden_lenght=5,output_lenght=3,fnet=fnet,df_dnet=df_dnet):
		self.model['input_lenght'] = input_lenght
		self.model['hidden_lenght'] = hidden_lenght
		self.model['output_lenght'] = output_lenght
		self.model['hidden_layer'] = np.random.uniform(-0.5,+0.5,(hidden_lenght,input_lenght+1))
		self.model['output_layer'] = np.random.uniform(-0.5,+0.5,(output_lenght,hidden_lenght+1))
		self.model['fnet'] = fnet
		self.model['df_dnet'] = df_dnet

	# Realiza o forward do algoritmo, onde calcula o output final a partir dos pesos atuais
	def forward(self, X):
		#Retirando os valores do modelo
		hidden = self.model['hidden_layer'] 
		output = self.model['output_layer']
		#Adicionando o 1 para a multiplicação
		X = np.concatenate((X,np.array([1])))

		#print('hidden=',hidden,'\n')
		#print('output=',output,'\n')
		
		#CAMADA ESCONDIDA
		net_h = np.matmul(hidden,X)
		#print('net_h=',net_h,'\n')
		f_net_h = self.model['fnet'](net_h)
		#print('f_net_h=',f_net_h,'\n')
		df_net_h = self.model['df_dnet'](f_net_h)
		#f_net_h = np.rint(f_net_h)
		
		#CAMADA SAÍDA
		f_net_h_c = np.concatenate((f_net_h,np.array([1])))
		net_o = np.matmul(output,f_net_h_c)
		#print('net_o=',net_o,'\n')
		#print('net_o=',net_o,'\n')
		f_net_o = self.model['fnet'](net_o)
		#print('f_net_o=',f_net_o,'\n')
		df_net_o = self.model['df_dnet'](f_net_o)
		#f_net_o = np.rint(f_net_o)

		return{
			"f_net_h": f_net_h,
			"df_net_h":df_net_h,
			"f_net_o": f_net_o,
			"df_net_o":df_net_o
		}

	# Realiza o treinamento da rede utilizando backpropagation com regra delta
	def backpropagation(self,X,Y,eta=0.5,max_error=0.000001,max_iter=500):
		counter = 0
		total_error = 2*max_error

		# Treinamento ocorre enquanto o erro for maior que o aceitavel ou o numero maximo de iterações nao tiver sido atingido
		while total_error > max_error and counter < max_iter:
			total_error = 0

			for i in range(0,X.shape[0]):
				x_i = X[i,:]
				y_i = Y[i]
				#print('x_i=',x_i,'\n')
				#print('y_i=',y_i,'\n')

				#forward
				fw = self.forward(x_i)

				#erro
				error_o_k = (y_i-fw['f_net_o'])
				#print('y_i',y_i,'\n')
				#print('f_net_o',fw['f_net_o'],'\n')
				#print('error_o_k',error_o_k,'\n')
				
				total_error = total_error + np.sum(error_o_k*error_o_k)

				#backpropagation / calculo das derivadas
				delta_out = error_o_k*fw['df_net_o']
				dE2_dw_o = np.multiply(np.array([-2*delta_out]).T,np.concatenate((fw['f_net_h'],np.array([1]))))

				delta_h = np.matmul(np.array([delta_out]),self.model['output_layer'][:,0:self.model['hidden_lenght']])
				dE2_dw_h = delta_h * (np.multiply(-2*fw['df_net_h'],np.array([np.concatenate((x_i,np.array([1])))]).T))

				# Atualização dos pesos
				#print('x_i.size=',x_i.size,'\n')
				#print('bag loko=',np.reshape(np.array([eta*dE2_dw_h]).T,(self.model['hidden_lenght'],x_i.size+1)),'\n')
				self.model['output_layer'] = self.model['output_layer'] - eta*dE2_dw_o
				self.model['hidden_layer'] = self.model['hidden_layer'] - np.reshape(np.array([eta*dE2_dw_h]).T,(self.model['hidden_lenght'],x_i.size+1))
				
			# Término da iteração do treinamento
			total_error = total_error/X.shape[0]
			counter = counter+1
			if (counter % 100) == 0:
				print("Iter:",counter," Error:",total_error)

		return

	def run(self,X,Y,size=2,eta=0.1,max_iter=500,train_size=0.7,threshold=0.000001):
		ids = random.sample(range(0,X.shape[0]),np.floor(train_size*X.shape[0]).astype(np.int))
		ids_left = diff(range(0,X.shape[0]),ids)
		#print('ids',ids,'\n')
		#print('ids_left',ids_left,'\n')

		# Training Set
		train_set = X[ids,:]
		train_classes = Y[ids,:]
		#print('X=',train_set)
		#print('Y=',train_classes)

		# Test Set
		test_set = X[ids_left,:]
		test_classes = Y[ids_left,:]
		#print('X=',test_set)
		#print('Y=',test_classes)

		self.architecture(input_lenght=X.shape[1],hidden_lenght=size,output_lenght=Y.shape[1])
		print('MLP architecture created\nStarting Training')
		self.backpropagation(train_set,train_classes,eta=eta,max_error=threshold,max_iter=max_iter)
		print('Neural Network Trained\nStarting Testing')
		#print(mlp.forward(X)['f_net_o'])

		correct = 0
		for i in range(0,test_set.shape[0]):
			x_i = test_set[i]
			y_i = test_classes[i]


			y_hat_i = np.round(self.forward(x_i)['f_net_o'])
			#print('y_hat_i',y_hat_i,'\n')
			if (np.sum((y_i - y_hat_i)**2) == 0):
				correct = correct + 1
			

			pass

		print('Neural Network Tested')
		accuracy = correct/test_set.shape[0]
		error = np.sum((y_i - y_hat_i)**2)/test_set.shape[0]
		
		return {
			"accuracy": accuracy,
			"error": error
		}
	#END OF MLP CLASS

# Reads the contents from a file and transforms in matrix
def matrix(contents):
	return [item.split(',') for item in contents.split('\n')[:-1]]

def class_ind(Y):
	unique_Y = set(Y)
	#print('unique_Y=',len(unique_Y),'\n')
	size = (Y.shape[0],len(unique_Y))
	res = np.zeros(size)
	for i in range(0,Y.shape[0]):
		res[i][Y[i].astype(np.int)-1] = 1

	#print('res=',res,'\n')
	return res

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def wine_test(eta=0.1,max_iter=500,train_size=0.7):
	for file in os.listdir():
		if(file.endswith('.data')):
			data = open(file).read()
			X = matrix(data)
			X = np.array(X)
			X = X.astype(np.float)
			Y = X[:,0]
			X = X[:,1:X.shape[1]]
			#print('X=',X)
			for i in range(X.shape[1]):
				X[:,i] = (X[:,i] - np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))
			#X = scale(X)
			Y = class_ind(Y)
			#print('X=',X)
			#print('Y=',Y)

	print('\nPreprocessing Wine Done')
	mlp = MLP()
	return mlp.run(X,Y,eta=eta,max_iter=max_iter,train_size=train_size)

def tracks_test(eta=0.1,max_iter=500,train_size=0.7):
	mlp = MLP()
	n = 1
	ret = {}
	for file in os.listdir():
		if(file.endswith('.txt')):
			data = open(file).read()
			X = matrix(data)
			X = np.array(X)
			X = X.astype(np.float)
			Y = X[:,X.shape[1]-2:X.shape[1]]
			X = X[:,0:X.shape[1]-2]
			for i in range(X.shape[1]):
				X[:,i] = (X[:,i] - np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))
			for i in range(Y.shape[1]):
				Y[:,i] = (Y[:,i] - np.amin(Y[:,i])) / (np.amax(Y[:,i]) - np.amin(Y[:,i]))
			#print('X=',X.shape[1])
			#print('Y=',Y.shape[1])
			print('\nPreprocessing Origin of Music Done')
			res = mlp.run(X,Y,eta=eta,max_iter=max_iter,train_size=train_size)
			#print('Error=',res['error'])
			if n == 1:
				ret['error_1'] = res['error']
			if n == 2:
				ret['error_2'] = res['error']
			n = n+1

	return ret


#################################################################################
# INICIO DO PROGRAMA															#
#################################################################################


print("Starting program...\n")
print("Choose database:\n\t1-Wine\n\t2-Geographical Original of Music\n")
op = input()
op = int(op)

if op == 1: # If wine database is chosen
	print('Wine Choosen')
	table = PrettyTable()
	table.field_names = ["Number of Cycles","Learning Speed","Training set size","Accuracy"]

	# Variation Learning Speed
	ret = wine_test(eta=0.1)
	table.add_row([500,0.1,0.7,ret['accuracy']])
	ret = wine_test(eta=0.3)
	table.add_row([500,0.3,0.7,ret['accuracy']])
	ret = wine_test(eta=0.5)
	table.add_row([500,0.5,0.7,ret['accuracy']])

	# Variating Number of Cycles
	ret = wine_test(max_iter=250)
	table.add_row([250,0.1,0.7,ret['accuracy']])
	ret = wine_test(max_iter=750)
	table.add_row([750,0.1,0.7,ret['accuracy']])
	ret = wine_test(max_iter=1000)
	table.add_row([1000,0.1,0.7,ret['accuracy']])

	# Variating Training set size
	ret = wine_test(train_size=0.5)
	table.add_row([500,0.1,0.5,ret['accuracy']])
	ret = wine_test(train_size=0.6)
	table.add_row([500,0.1,0.6,ret['accuracy']])
	ret = wine_test(train_size=0.9)
	table.add_row([500,0.1,0.9,ret['accuracy']])

	# Printing Results
	print(table)

elif op == 2:
	print('Origin of Music Choosen')
	table = PrettyTable()
	table.field_names = ["Number of Cycles","Learning Speed","Training set size","First file Mean Square Error","Second file Mean Square Error"]

	#Variating Learning Speed
	err = tracks_test(eta=0.1)
	table.add_row([500,0.1,0.7,err['error_1'],err['error_2']])
	err = tracks_test(eta=0.3)
	table.add_row([500,0.3,0.7,err['error_1'],err['error_2']])
	err = tracks_test(eta=0.5)
	table.add_row([500,0.5,0.7,err['error_1'],err['error_2']])

	#Variating Number of Cycles
	err = tracks_test(max_iter=300)
	table.add_row([300,0.1,0.7,err['error_1'],err['error_2']])
	err = tracks_test(max_iter=500)
	table.add_row([700,0.1,0.7,err['error_1'],err['error_2']])
	err = tracks_test(max_iter=700)
	table.add_row([1000,0.1,0.7,err['error_1'],err['error_2']])

	#Variating Training Set Size
	err = tracks_test(train_size=0.5)
	table.add_row([500,0.1,0.5,err['error_1'],err['error_2']])
	err = tracks_test(train_size=0.75)
	table.add_row([500,0.1,0.75,err['error_1'],err['error_2']])
	err = tracks_test(train_size=0.9)
	table.add_row([500,0.1,0.9,err['error_1'],err['error_2']])

	print(table)