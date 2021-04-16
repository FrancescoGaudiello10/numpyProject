# Video tutorial Edureka - NumPy:
# https://www.youtube.com/watch?v=8JfDAm9y_7s&list=PLzoetWSUagaa8R9xdaBifysYwgQH0KBTZ&index=1

import numpy as np
import time
import sys
import matplotlib.pyplot as plt

# Array monodimensionale
print("Array A:")
a = np.array([1, 2, 3])
print(a)

# Array bidimensionale
print("Array B")
b = np.array([(1, 2, 3),
              (4, 5, 6)])
print(b)
print("*****************")

# Vantaggi numPy = utilizza molta meno memoria (1)
S = range(1000)
print("Utilizzo di memoria:")
print("Getsizeof (Python Standard) = ", sys.getsizeof(5)*len(S))

D = np.arange(1000)
print("With arange (numpy) = ", D.size*D.itemsize)
print("*****************")

# Memoria predefinita
SIZE = 1000000
L1 = range(SIZE)
L2 = range(SIZE)

A1 = np.arange(SIZE)
A2 = np.arange(SIZE)

start = time.time()
# Modo alternativo per cerare una lista con un ciclo
# Si possono effettuare molte operazioni con NumPy
result = [(x, y) for x, y in zip(L1, L2)]
print("Time: ", (time.time() - start) * 1000)

# Find the n-dimension of the array
c = np.array([(1, 2, 3), (5, 6, 7)])
print("ndim =", c.ndim)
# Length of one array element in bytes
print("itemsize =", c.itemsize)
# Return a data type
print("dtype =", c.dtype)
print("*****************")

cc = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("size =", cc.size)
# shape = Return the shape of an array.
print("shape =", cc.shape)  # (10,)
print("*****************")

d = np.array([(1, 2, 3),
              (4, 7, 9)])
print("size =", d.size)
print("shape =", d.shape)   # (2,3) -> righe-colonne
print("*****************")

# Reshape = Gives a new shape to an array without changing its data.
e = np.array([(1, 2, 3, 4),
              (5, 6, 7, 8)])
print("Before reshape =\n", e)
e = e.reshape(4, 2)
print("After reshape =\n", e)
print("*****************")

f = np.array([(1, 2, 3, 4),
              (5, 6, 7, 8),
              (9, 10, 11, 12)])
print("Take a element")
print("f[0,2] =", f[0, 2])
print("Take all index from 0 and only columns 3")
print("f[0:,3] =", f[0:, 3])
print("Take from index 0 to 2 and only columns 3")
print("f[0:2, 3] =", f[0:2, 3])     # index 2 not incluse
print("*****************")

# linspace = restituisce numeri equidistanti su un intervallo specificato.
print("linspace: restituisce numeri equidistanti su un intervallo specificato")
g = np.linspace(1, 3, 5)
print("linspace(1,3,5) =", g)
g2 = np.linspace(1, 3, 10)
print("linspace(1,3,10) =", g2)
print("*****************")

print(" Max - Min - Sum")
h = np.array([1, 3, 5, 6, 7, 8])
print("h.max=", h.max())
print("h.min=", h.min())
print("h.sum=", h.sum())

# axis=0 -> tutte le colonne tra di loro
# axis=1 -> tutte le righe tra di loro
i = np.array([(1, 0, 8),
              (9, 5, 3)])
print("Somma tutte le colonne: i.sum(axis=0)=", i.sum(axis=0))
print("Somma tutte le righe: i.sum(axis=1)=", i.sum(axis=1))
print("*****************")

k = np.array([(1, 3, 49),
              (25, 9, 16)])
print("Sqrt:\n", np.sqrt(k))
print("Deviazione Standard =", np.std(k))
print("*****************")

# Operazioni addizione, sottrazione, moltiplicazione, divisione (con matrici)
m = np.array([(1, 2, 4),
              (5, 9, 10)])
n = np.array([(6, 4, 3),
              (7, 11, 7)])
print("Addizione (m+n):\n", m+n)
print("Sottrazione (m-n):\n", m-n)
print("Moltiplicazioen (m*n):\n", m*n)
print("Divisione (m/n):\n", m/n)
print("*****************")

# vstack e hstack
print("np.vstack: \n", np.vstack((m, n)))
print("np.hstack: \n", np.hstack((m, n)))
print("*****************")

# ravel() = Restituisce un array appiattito contiguo.
# Rif: https://numpy.org/doc/stable/reference/generated/numpy.ravel.html?highlight=ravel#numpy.ravel
print("Ravel:\n", m.ravel())
print("*****************")

# Utilizzo libreria MatPlotLib per creare grafici.
# arange = Restituisce valori equidistanti all'interno di un dato intervallo.
xx = np.arange(0, 3*np.pi, 0.1)
yy = np.sin(xx)
plt.plot(xx, yy)
# plt.show()
zz = np.tan(xx)
plt.plot(xx, zz)
# plt.show()

# Esponenziale e logaritmica
ab = np.array([1, 2, 3])
print("exp: ", np.exp(ab))
print("log: ", np.log(ab))
