import numpy as np 
import numpy.linalg as la ##para autovetor e autovalor 
import pandas as pd
from sklearn.preprocessing import StandardScaler ### para normalizar os dados
import matplotlib.pyplot as plt



scaler = StandardScaler()

df_original = pd.read_csv('bd.csv', names=['sepal length','sepal width','petal length','petal width','target'])

df = pd.read_csv('bd.csv', names=['sepal length','sepal width','petal length','petal width','target'])
print('iris original:','\n' ,df) 

print(df.loc[0,"sepal length"])
del df['target'] ##apagando a coluna de strings

scaler.fit(df)
iris = scaler.transform(df) #####Normaliza os dados
print("dados transformado:",'\n',iris,'\n')

mat_cov = np.cov(np.transpose(iris)) ###### Matriz de covariância
print("matriz de covariância:","\n",mat_cov, "\n")

val,vet=la.eig(mat_cov)
print("Autovalores: ","\n", val)
print("Autovetores:","\n", vet)


soma = sum(vet)

vet = vet[:,0:2]
print("new vet=",np.transpose(vet),"\n")

new_data = np.transpose(np.matmul(np.transpose(vet),np.transpose(iris)))

classes = np.zeros(iris.shape[0])
for i in range(0,iris.shape[0]):
	if df_original['target'][i] == 'Iris-setosa':
		classes[i] = 1
	elif df_original['target'][i] == 'Iris-versicolor':
		classes[i] = 2
	elif df_original['target'][i] == 'Iris-virginica':
		classes[i] = 3


new_data = np.insert(new_data,new_data.shape[1],classes,axis=1)


plt.xlabel('Z1 Component')
plt.ylabel('Z2 Component')
plt.title("Data with PCA analysis")
plt.scatter(new_data[:,0],new_data[:,1],c=new_data[:,2],alpha=0.8)
plt.show()
