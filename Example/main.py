# Video tutorial Edureka - NumPy:
# https://www.youtube.com/watch?v=8JfDAm9y_7s&list=PLzoetWSUagaa8R9xdaBifysYwgQH0KBTZ&index=1

import numpy as np
import time
import sys
import matplotlib.pyplot as plt

# Esempio Array standard monodimensionale
a = np.array([1,2,3])
print(a)
print("********")

# Esempio Array bidimensionale
aa = np.array([(1,2,3), (4,5,6)])
print(aa)
print("********")

# Esempio - Convenienza di NumPy 1: utilizza molta meno memoria
S = range(1000)
print(sys.getsizeof(5)*len(S))

D = np.arange(1000)
print(D.size*D.itemsize)
print("********")

# Esempio com memoria predefinita
SIZE = 1000000
L1 = range(SIZE)
L2 = range(SIZE)

A1 = np.arange(SIZE)
A2 = np.arange(SIZE)

start = time.time()
result = [(x,y) for x,y in zip(L1, L2)] # Modo alternativo per creare una lista con un ciclo in un unica riga
print((time.time() - start)*1000)

start = time.time()
result = A1+A2
print((time.time() - start)*1000)
print("********")
# Noto i due risultati che effettuano la medesima operazione. Utilizzando NumPy e piu efficiente.
# Si possono effettuare molte operazioni con NumPy

# Find the dimension of the array:
b = np.array([(1,2,3), (4,5,6)])
print("ndim:", b.ndim) #Ritorna la dimensione n-esima dell'array.
print("itemsize:", b.itemsize)
print("dtype:", b.dtype)
print("********")

c = np.array([(1,2,3,4,5,6,7,8)])
print("size:", c.size)
print("shape:", c.shape)

d = np.array([(1,2,8),
              (4,5,9)])
print("size:", d.size)
print("shape:", d.shape)
print("********")

k = np.array([(1,2,3,4),(5,6,7,8)])
print("before reshape:", k)
k = k.reshape(4,2)
print("after reshape:", k)
print("********")

kk = np.array([(1,2,3,4),(5,6,7,8),(7,10,12,14)])
print(kk[0,2]) #Prende l'elemento in posizione 0,2 che corrisponde a "3"
print(kk[0:,3]) #Parte dalla riga 0 e prende tutti gli elementi sotto la colonna 3
print(kk[0:2,3]) #Prende dalla riga 0 alla riga 2 tutti gli elementi sotto la terza colonna
print("********")

# linspace = Parte da 1 fino a 3 con intervalli di 1/2
p = np.linspace(1,3,5)
print(p)
p = np.linspace(1,3,10)
print(p)
print("********")

# Funzione Max
z =  np.array([1,3,5,10,7])
print(z.max())
print(z.min())
print(z.sum())
print("********")

# axis = 0 -> somma tutte le colonne tra di loro
# axis = 1 -> somma tutte le righe tra di loro
zz = np.array([(1,0,7),(6,10,23)])
print("Somma delle colonne: ", zz.sum(axis=0))
print("Somma delle righe: ",zz.sum(axis=1))
print("********")

#Radice quadrata e deviazione
ar = np.array([(1,0,7),(6,10,16)])
print("sqrt: ", np.sqrt(ar)) #sqrt di ogni elemento
print("deviaz: ", np.std(ar)) #Deviazione standard
print("********")

#Operazioni standard di addizione, sottrazione, moltiplicazione, divisione con le matrici
vi = np.array([(1,2,3),
               (4,5,6)])
vu = np.array([(1,2,3),
               (4,5,6)])
print("Addizione matrici:\n", vi+vu)
print("Sottrazione matrici:\n", vi-vu)
print("Moltiplicazione matrici:\n", vi*vu)
print("Divisioni matrici:\n", vi/vu)
print("***********")

print(np.vstack((vi,vu))) #Unisce le due matrici con righe sotto
print(np.hstack((vi,vu))) #Unisce le due matrici con colonne a destra
print("***********")

print(vi.ravel()) #metodo che fa diventare un array MONODIMENSIONALE una matrice
print("***********")

# Utilizzo della libreria MATPLOTLIB per creare grafici.
print("**SPECIAL FUNCTIONS**")
xx = np.arange(0, 3*np.pi, 0.1)
yy = np.sin(xx)
plt.plot(xx,yy)
#plt.show()
zz = np.tan(xx)
plt.plot(xx,zz)
#plt.show()

# Funzione esponenziale e logaritmica
ap = np.array([1,2,3])
print("exp: ", np.exp(ap))
print("log: ", np.log(ap))

print("****************** Secondo Sprint ********************")

print("*** THE BASICS ***")
a = np.array([1,2,3,4])
print(a)

b = np.array([(1,2,3,4,5),(6,7,8,9,10)])
print(b)

#Get Dimension
print(a.ndim)

#Get Shape
print(a.shape)
print(b.shape)

#Get Type
print(a.dtype)
print(b.dtype)

#Get Size
print(a.size)
print(a.itemsize) #1 int = 4 byte
print(b.itemsize)
print(a.nbytes)
print("** END **")

print("** ACCESSING SPECIFIC ELEMENT ECC.. **")
x = np.array([(1,2,3,4,5,6,7),
              (10,11,12,14,15,16,19)])
print(x)

# Get a specific element [r, c]
print(x[1,5]) #Riga 1 - Colonna 5

# Get a specific row
print(x[0,:])

# Get a specific column
print(x[:, 1])

# Getting a little more fancy [startindex:endindex:stepsize]
print(x[0,1:6]) #Prendo tutta la riga 0 dalla colonna 1 alla 6
print(x[0,1:6:2]) #Prendo tutta la riga 0 dalla colonna 1 alla 6 con PASSO di 2
print("*****")

print("** INIZIALIZING DIFFERENTE TYPES OF ARRAYS **")
# ALL 0s matrix
zr = np.zeros(2)
print(zr)
zr2 = np.zeros((2,3,3,2))
print(zr2)

# Due matrici -> Due righe e Tre colonne
print("** ALL 1s matrix **")
one = np.ones((2,2,3))
print(one)

# Any other number
# Array bi-dimensionale 2 righe x 2 colonne tutte con 99 come elementi
full = np.full((2,2), 99)
print(full)

print("** Random number **")
ran = np.random.rand(5,2,3)
print(ran)
ran2 = np.random.random_sample(x.shape)
print(ran2)

print("*** Random Integer***")
randInt = np.random.randint(-5,10, size=(3,3)) #Genera una matrice random 3x3
print(randInt)

print(np.identity(5)) # Stampa una matrice identita
print("******")

narr = np.array([1,2,3])
r1 = np.repeat(narr, 3, axis=0)
print(r1)

print("*******")
output = np.ones((5,5))
print(output)
zero = np.zeros((3,3))
print(zero)
zero[1,1] = 9
print(zero)

output[1:4, 1:4] = zero
print(output)

print("***************************************")
print("*** Example Mathematics Operations ***")
print("***************************************")
print("Array iniziale")
mat1 = np.array([1,2,3,4,5])
print(mat1)
print("Addizione")
mat1 = mat1+2
print(mat1)
print("Sottrazione")
mat1 = mat1-3
print(mat1)
print("Moltiplicazione")
mat1 = mat1*4
print(mat1)
print("Divisione")
mat1 = mat1/2
print(mat1)

mat2 = np.array([1,0,1,0,1])
mat2 = mat1+mat2
print(mat2)
mat2 = mat2**2 #Elevamento potenza
print(mat2)

print("*** Take the sin ***")
sin = np.sin(mat2)
print(sin)
print("*** Take the cos ***")
cos = np.cos(mat2)
print(cos)

# Find the determinant
print("Find the determinant:")
c = np.identity(3)
rs = np.linalg.det(c)
print(rs)

#Statistics
stats = np.array([[1,2,3],
                  [4,5,6]])
print(stats)
mnr = np.min(stats, axis=1) #Stampa i minimi di ogni riga
mnc = np.min(stats, axis=0) #Stampa i minimi di ogni riga
print("min di ogni riga:", mnr)
print("min di ogni col:", mnc)
mx = np.max(stats) #stampa il massimo in generale
print("max:", mx)

nsum = np.sum(stats,axis=0)
print("sum ogni colonna:", nsum)
nsumr = np.sum(stats, axis=1)
print("sum ogni riga:", nsumr)
sumgen = np.sum(stats)
print("sum totale:", sumgen)


print("*** REORGANIZING ARRAYS ***")
before = np.array([[1,2,3,4],[5,6,7,8]])
print(before)
print("shape:\n", before.shape)

after = before.reshape((4,2))
print(after)

print("*** Vertical stacking vectors ***")
v1 = np.array([1,2,3,4])
v2 = np.array([4,5,6,7])
print(np.vstack([v1,v2,v2,v1]))

print("*** Horizontal stacking vectors ***")
print(np.hstack([v1,v2,v2,v1]))

### Miscellaneous
print("*********************")
print("*** Miscellaneous ***")
print("*********************")

ms = np.genfromtxt('data.txt', delimiter=',')
print(ms)

# Advanced Boolean Masking indexing
print(ms>30) #return true or false
print(ms[ms > 20]) #return il valore di tutti i > 20 sotto forma di array.

# You can index with a list in NumPY
aaa = np.array([1,2,3,4,5,6,7,8,9])
print(aaa[[1,2,8]]) #ritorna gli elementi in indice 1,2,8

