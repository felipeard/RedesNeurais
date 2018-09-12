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
		return (1 / (1 + exp(-net)))

	def df_dnet(f_net):
		return (f_net * (1 - f_net))

	def architecture(self,input_lenght=10,hidden_lenght=np.floor(np.log2(10)).astype(np.int),output_lenght=10,fnet=fnet,df_dnet=df_dnet):
		self.model['input_lenght'] = input_lenght
		self.model['hidden_lenght'] = hidden_lenght
		self.model['output_lenght'] = output_lenght
		self.model['hidden_layer'] = np.random.uniform(-0.5,+0.5,(hidden_lenght,input_lenght+1))
		self.model['output_layer'] = np.random.uniform(-0.5,+0.5,(output_lenght,hidden_lenght+1))
		self.model['fnet'] = fnet
		self.model['df_dnet'] = df_dnet


	def forward(self):
		return

	def backpropagation(self,X,Y,eta=0.1,max_error=0.0000001,max_iter=2000):
		counter = 0
		error = 2*max_error

		while total_error > max_error and counter < max_iter:
			error = 0
			

		return

mlp = MLP()
print(mlp.model)