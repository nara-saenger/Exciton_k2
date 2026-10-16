import pandas as pd
import numpy as np

dados = pd.read_csv("IT-4F-dipolos.csv")

# dipolos 

x_dipolo = dados["X"].to_numpy()
y_dipolo = dados["Y"].to_numpy()
z_dipolo = dados["Z"].to_numpy()
norm = dados["Abs_mi"]

x_dipolo = x_dipolo/norm
y_dipolo = x_dipolo/norm
z_dipolo = x_dipolo/norm

dipoles = []
for i in range(len(x_dipolo)):
    dipoles.append([round(x_dipolo[i],5),round(y_dipolo[i],5),round(z_dipolo[i],5)]) 
print(dipoles)

np.savetxt('dipoles.txt',np.array(dipoles))

#posições-rede-lattice

x = dados["CM_X"].to_numpy()
y = dados["CM_Y"].to_numpy()
z = dados["CM_Z"].to_numpy()

latt = []
for i in range(len(x_dipolo)):
    latt.append([round(x[i],5),round(y[i],5),round(z[i],5)]) 
print(latt)

np.savetxt('lattice.txt',np.array(latt))

