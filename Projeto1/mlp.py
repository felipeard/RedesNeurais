import numpy as np

net = input("net Value:")
net = np.exp(net)
f = 1/(1+net)
print(f)