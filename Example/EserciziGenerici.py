# Rif: https://numpy.org/doc/stable/user/quickstart.html
import numpy as np

# arange = crea un array da 0 a n
# reshape effettua il re formato righe-colonne
a = np.arange(15).reshape(3, 5)
print("Array a:\n", a)

# restituisce la dimensione
print(a.shape)

# restituisce la n-dimensione (r, c) in questo caso 2
print(a.ndim)

# restituisce il numero di elementi
print(a.size)

# restituisce il type
print(a.dtype.name)

# restitusce la lunghezza in byte di ogni elemento dell'array (int64 = 8)
print(a.itemsize)

# creazione standard array
b = np.array([1, 2, 3, 4, 5])
print("Array b:\n", b)

# array bidimensionale
arr_2 = np.array([[1, 2, 3, 4], [5, 7, 8, 10]])
print("Array bi-dimensionale:\n", arr_2)

# Array di zeri o uni
z = np.zeros((3, 4))
print(z)

o = np.ones((4, 5))
print(o)

lin = np.linspace(0, 3, 9)
print("lin:\n", lin)

# Sono due matrici, ognuna con 3 righe e 4 colonne
arr_3_dim = np.arange(24).reshape(2, 3, 4)
print("array_3_dim:\n", arr_3_dim)

# Prodotto matriciale
# case 1
m = np.array([[1, 1],
              [0, 1]])
n = np.array([[2, 0],
              [3, 4]])

print(m.dot(n))

# case 2
print(m @ n)

print(m.sum())
print(n.sum())

print(m.sum(axis=0))    # colonna
print(n.sum(axis=0))

print(m.sum(axis=1))    # righe
print(n.sum(axis=1))

# Array monodimensionale: One-dimensional arrays can be indexed,
# sliced and iterated over, much like lists and other Python sequences.
sl = np.arange(10) ** 3
print(sl)
print(sl[4])
print(sl[2:6])      # Stampa da 2 incluso a 6 escluso
print(sl[:7:3])     # stampa tutto fino a index 7, con passo 3
print(sl[::-1])     # stampa array reverse


# Multidimensional arrays can have one index per axis.
# These indices are given in a tuple separated by commas.
def f(x, y):
    return 10 * x + y


multi = np.fromfunction(f, (5, 4), dtype=int)
print("\n", multi)
print(multi[2, 3])
print(multi[0:4, 1])    # stampa da 0 a 4 indice riga della colonna index 1
print(multi[:, 2])      # stampa tutte le righe della colonna index 2
print(multi[1:3, :])    # stampa righe da 1 a 3 esclusa, tutte colonne

print(multi[-1])        # stampa l'ultima riga


c = np.array([[[1, 2, 3],
              [4, 5, 6]],
              [[7, 8, 10],
               [12, 14, 5]]])
print("c.shape:\n", c.shape)
print(c[1, ...])    # stampa la seconda matrice
print(c[..., 2])    # stampa la colonna index 2 di tutte le matrici

# ciclo for
for row in b:
    print(row)

print("*** Shape Manipulation ***")
a = np.floor(10*np.random.randn(3, 4))  # tutti float
print(a)
print(a.shape)
print("changing shape...")
print(a.ravel())    # return array flattened
print(a.T)          # return array transposted
print(a.T.shape)

print("array a:\n", a)
a.resize((2, 6))
print("array after resize:\n", a)