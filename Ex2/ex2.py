# Exercicio 2 da matéria de Redes Neurais
#	Felipe Alegria Rollo Dias	nUSP 9293501
#	Leonardo Piccaro Rezende	nUSP 9364611
#

import numpy as np

class MLP(object):

	model = {}
	"""docstring for MLP"""
	def __init__(self):
		# Iniatialize MLP class
		self.architecture()

	def fnet(net):
		return (1 / (1 + np.exp(-net)))

	def df_dnet(f_net):
		return (f_net * (1 - f_net))

	def architecture(self,input_lenght=10*10,hidden_lenght=np.floor(np.log2(10*10)).astype(np.int),output_lenght=10*10,fnet=fnet,df_dnet=df_dnet):
		self.model['input_lenght'] = input_lenght
		self.model['hidden_lenght'] = hidden_lenght
		self.model['output_lenght'] = output_lenght
		self.model['hidden_layer'] = np.random.uniform(-0.5,+0.5,(hidden_lenght,input_lenght+1))
		self.model['output_layer'] = np.random.uniform(-0.5,+0.5,(output_lenght,hidden_lenght+1))
		self.model['fnet'] = fnet
		self.model['df_dnet'] = df_dnet


	def forward(self, X):
		#Retirando os valores do modelo
		hidden = self.model['hidden_layer'] 
		output = self.model['output_layer']
		#Adicionando o 1 para a multiplicação
		X = np.concatenate((X,np.array([1])))
		
		#CAMADA ESCONDIDA
		net_h = np.matmul(hidden,X)
		f_net_h = self.model['fnet'](net_h)
		df_net_h = self.model['df_dnet'](f_net_h)
		f_net_h = np.rint(f_net_h)
		
		#CAMADA SAÍDA
		f_net_h = np.concatenate((f_net_h,np.array([1])))
		net_o = np.matmul(output,f_net_h)
		f_net_o = self.model['fnet'](net_o)
		df_net_o = self.model['df_dnet'](f_net_o)
		f_net_o = np.rint(f_net_o)

		return{
			"f_net_h": f_net_h,
			"df_net_h":df_net_h,
			"f_net_o": f_net_o,
			"df_net_o":df_net_o
		}

	def backpropagation(self,X,Y,eta=0.1,max_error=0.0000001,max_iter=2000):
		counter = 0
		error = 2*max_error

		while total_error > max_error and counter < max_iter:
			error = 0

			#forward

			#erro

			#backpropagation

		return

N = 10 
X = np.identity(N)
X = np.reshape(X,N*N)
Y = np.identity(N)
Y = np.reshape(Y,N*N)
mlp = MLP()
#print(mlp.model)
print(mlp.forward(X))