# Exercicio 2 da matéria de Redes Neurais
#	Felipe Alegria Rollo Dias	nUSP 9293501
#	Leonardo Piccaro Rezende	nUSP 9364611
#

import numpy as np
import os
from sklearn.preprocessing import scale

# Classe que representa uma MLP com sua propria arquitetura
class MLP(object):

	# Dictionary do python para armazenar o modelo da MLP
	model = {}
	# Construtor da classe
	def __init__(self):
		# Iniatialize MLP class
		self.architecture()

	# Função de Ativação dos neuronios da MLP
	def fnet(net):
		return (1 / (1 + np.exp(-net)))

	# Função para calcula a derivada
	def df_dnet(f_net):
		return (f_net * (1 - f_net))

	# Função que inicializa a arquitetura da MLP baseado no problema especifico
	def architecture(self,input_lenght=13,hidden_lenght=5,output_lenght=1,fnet=fnet,df_dnet=df_dnet):
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
		f_net_o = self.model['fnet'](net_o)
		#print('f_net_o=',f_net_o,'\n')
		df_net_o = self.model['df_dnet'](f_net_o)
		f_net_o = np.rint(f_net_o)

		return{
			"f_net_h": f_net_h,
			"df_net_h":df_net_h,
			"f_net_o": f_net_o,
			"df_net_o":df_net_o
		}

	# Realiza o treinamento da rede utilizando backpropagation com regra delta
	def backpropagation(self,X,Y,eta=0.5,max_error=0.0000001,max_iter=5):
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
				#print('f_net_o=',fw['f_net_o'],'\n')
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
			print("Iter:",counter," Error:",total_error,"\n")

		return

# Reads the contents from a file and transforms in matrix
def matrix(contents):
	return [item.split(',') for item in contents.split('\n')[:-1]]

#################################################################################
# INICIO DO PROGRAMA															#
#################################################################################


print("Starting program...\n")
print("Choose database:\n\t1-Wine.data\n")
if(1): # If wine database is chosen
	for file in os.listdir():
		if(file.endswith('.data')):
			data = open(file).read()
			X = matrix(data)
			X = np.array(X)
			X = X.astype(np.float)
			Y = X[:,0]
			X = X[:,1:X.shape[1]]
			print('X=',X)
			for i in range(X.shape[1]):
				X[:,i] = (X[:,i] - np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))
			#X = scale(X)
			print('X=',X)
			#print('Y=',Y)

mlp = MLP()
mlp.backpropagation(X,Y)
#print(mlp.forward(X)['f_net_o'])