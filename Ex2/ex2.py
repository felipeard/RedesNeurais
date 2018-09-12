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

	def architecture(self,input_lenght=10,hidden_lenght=np.log2(10),output_lenght=10,fnet=fnet,df_dnet=df_dnet):
		self.model['input_lenght'] = input_lenght
		self.model['hidden_lenght'] = hidden_lenght
		self.model['output_lenght'] = output_lenght
		self.model['fnet'] = fnet
		self.model['df_dnet'] = df_dnet


	def forward(self):
		return

	def backpropagation(self):
		return

mlp = MLP()
print(mlp.model)