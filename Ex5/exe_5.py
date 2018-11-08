import numpy as np 
import numpy.linalg as la ##para autovetor e autovalor 
import pandas as pd
from sklearn.preprocessing import StandardScaler ### para normalizar os dados
scaler = StandardScaler()
import matplotlib.pyplot as plt

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


