# Exercicio 2 da matéria de Redes Neurais
#	Felipe Alegria Rollo Dias	nUSP 9293501
#	Leonardo Piccaro Rezende	nUSP 9364611
#

import numpy as np

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
	def architecture(self,input_lenght=10*10,hidden_lenght=np.floor(np.log2(10*10)).astype(np.int),output_lenght=10*10,fnet=fnet,df_dnet=df_dnet):
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
		
		#CAMADA ESCONDIDA
		net_h = np.matmul(hidden,X)
		f_net_h = self.model['fnet'](net_h)
		df_net_h = self.model['df_dnet'](f_net_h)
		f_net_h = np.rint(f_net_h)
		
		#CAMADA SAÍDA
		f_net_h_c = np.concatenate((f_net_h,np.array([1])))
		net_o = np.matmul(output,f_net_h_c)
		f_net_o = self.model['fnet'](net_o)
		df_net_o = self.model['df_dnet'](f_net_o)
		f_net_o = np.rint(f_net_o)

		return{
			"f_net_h": f_net_h,
			"df_net_h":df_net_h,
			"f_net_o": f_net_o,
			"df_net_o":df_net_o
		}

	# Realiza o treinamento da rede utilizando backpropagation com regra delta
	def backpropagation(self,X,Y,eta=0.1,max_error=0.0000001,max_iter=2000):
		counter = 0
		total_error = 2*max_error

		# Treinamento ocorre enquanto o erro for maior que o aceitavel ou o numero maximo de iterações nao tiver sido atingido
		while total_error > max_error and counter < max_iter:
			total_error = 0

			#forward
			fw = self.forward(X)

			#erro
			error_o_k = (Y-fw['f_net_o'])
			total_error = total_error + np.sum(error_o_k*error_o_k)

			#backpropagation / calculo das derivadas
			delta_out = error_o_k*fw['df_net_o']
			dE2_dw_o = np.multiply(np.array([-2*delta_out]).T,np.concatenate((fw['f_net_h'],np.array([1]))))

			delta_h = np.matmul(np.array([delta_out]),self.model['output_layer'][:,0:self.model['hidden_lenght']])
			dE2_dw_h = delta_h * (np.multiply(-2*fw['df_net_h'],np.array([np.concatenate((X,np.array([1])))]).T))

			# Atualização dos pesos
			self.model['output_layer'] = self.model['output_layer'] - eta*dE2_dw_o
			self.model['hidden_layer'] = self.model['hidden_layer'] - np.reshape(np.array([eta*dE2_dw_h]).T,(self.model['hidden_lenght'],X.size+1))
			
			# Término da iteração do treinamento
			counter = counter+1
			print("Iter:",counter," Error:",total_error,"\n")

		return

# INICIO DO PROGRAMA

N = 10 #Tamanho da matriz identide
X = np.identity(N) # Cria a entrada
X = np.reshape(X,N*N) # Transforma a matriz para um vetor
Y = np.identity(N) # Cria a saida
Y = np.reshape(Y,N*N)
mlp = MLP() # cria um objeto da classe MLP para rodarmos o algoritimo
mlp.backpropagation(X,Y) # realiza o treinamento da rede

# Imprime o resultado do teste da rede final
print(np.reshape(mlp.forward(X)['f_net_o'],(N,N)))