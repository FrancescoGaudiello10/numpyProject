# NumPy FreeCodeCamp
# Rif: https://www.youtube.com/watch?v=QUT1VHiLmmI&list=PLzoetWSUagaa8R9xdaBifysYwgQH0KBTZ&index=2

import numpy as np
import time
import sys
import matplotlib.pyplot as plt

a = np.array([1, 2, 3, 4, 5])
b = np.array([(1, 2, 3, 4, 5),
              (6, 7, 9, 10, 12)])
print("Array A:\n", a)
print("Array B:\n", b)

print("a.n-dim:", a.ndim)
print("a.shape:", a.shape)
print("b.n-dim:", b.ndim)
print("b.shape:", b.shape)
print("a.dtype:", a.dtype)
print("b.dtype:", b.dtype)
print("a.size:", a.size)
print("b.size", b.size)
print("a.itemsize", a.itemsize)   # 1 int = 4 byte
print("b.itemsize", b.itemsize)
print("a.nbytes", a.nbytes)
print("b.nbytes", b.nbytes)
print("******************")

print("*** Accessing specific element and simile ***")
x = np.array([(1, 2, 3, 4, 5, 6, 7),
              (7, 8, 9, 10, 11, 12, 13)])
print("Array:\n", x)

# Get a specific element [r, c]
print("x[1,5]:", x[1, 5])
# Get a specific row - all columns
print("x[0,:]:", x[0, :])
# Get a specific column - all rows
print("x[:, 1]:", x[:, 1])

# Get row 0 and columns from 1 to 6
print("x[0, 1:6]:", x[0, 1:6])
# Get row 0 and columns from 1 to 6 by pass 2
print("x[0, 1:6:2]:", x[0, 1:6:2])
print("******************")

print("*** Inizializing different types of arrays ***")
# ALL 0s matrix
zr = np.zeros(2)
print("np.zeros(2):", zr)
zr2 = np.zeros((2, 3, 3, 2))
print("np.zeros((2, 3, 3, 2)):\n", zr2)

# 2 matrix -> 2 rows 3 columns
print("*** All 1s matrix ***")
one = np.ones((2, 2, 3))
print(one)

# Any other numbers
# Array 2-dim with 2 rows and 2 columns wiht every "99"
full = np.full((2,2), 99)
print("Full (99):\n", full)

print(" *** Random Numbers ***")
ran = np.random.rand(5, 2, 3)
print("Random:\n", ran)
ran2 = np.random.random_sample(x.shape)
print("\nRandom_sample(x.shape):\n", ran2)

print("*** Random Integer ***")
print("Genera una matrice 3x3...")
randInt = np.random.randint(-5, 10, size=(3, 3))
print(randInt)

print("\nIdentity Matrix")
print(np.identity(5))
print("******************")

narr = np.array([1,2,3])
r1 = np.repeat(narr, 3, axis=0)
print("Repeat:\n", r1)

out = np.ones((5, 5))
print("Ones:\n", out)
zero = np.zeros((3, 3))
print("Zeros:\n", zero)

zero[1, 1] = 9
print("Zero aggiornato:\n", zero)

out[1:4, 1:4] = zero
print("Out aggiornato:\n", out)
print("******************")

print("*** Example Mathematics Operations ***")
mat1 = np.array([1, 2, 3, 4, 5])
print("Array start:\n", mat1)
print("Addizione+2:\n", mat1+2)
print("Sottrazione-3:\n", mat1-3)
print("Moltiplicazione*4:\n", mat1*4)
print("Divisione/2:\n", mat1/2)

mat2 = np.array([1, 0, 1, 0, 1])
mat2 = mat1+mat1
print("Somma di due matrici:\n", mat2)

# Elevamento a potenza
mat2 = mat2**mat2
print("Elevamento a potenza (mat2**mat2):\n", mat2)
print("******************")

print("*** Take Sin/Cos ***")
sin = np.sin(mat2)
print("Sin: ", sin)
cos = np.cos(mat2)
print("Cos: ", cos)

# Find the determiant
print("Find the determinant..")
c = np.identity(4)
d = np.linalg.det(c)
print("np.linalg.det(c): ", d)

# Statistics
stats = np.array([(1, 2, 3),
                  (4, 5, 6)])
print("Matrice:\n", stats)
mnr = np.min(stats, axis=1)
print("Minimo di ogni riga:\n", mnr)
mnc = np.min(stats, axis=0)
print("Minimo di ogni colonna:\n", mnc)

mxr = np.max(stats, axis=1)
print("Massimo di ogni riga:\n", mxr)
mxc = np.max(stats, axis=0)
print("Massimo di ogni colonna:\n", mxc)

print("Massimo di tutta la matrice:\n", stats.max())

nsumc = np.sum(stats, axis=0)
print("Somma di ogni colonna:\n", nsumc)
nsumr = np.sum(stats, axis=1)
print("Somma di ogni riga:\n", nsumr)

print("Somma di tutti gli elementi:\n", stats.sum())
print("******************")

print("*** Reorganizing Arrays ***")
before = np.array([(1, 2, 3, 4),
                   (5, 6, 7, 9)])
print("Before:\n", before)
print("Shape: (print dimension)\n", before.shape)

after = before.reshape((4, 2))
print("After Reshaping:\n", after)

print("******************")
print("*** Vertical Stacking Vectors ***")
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 9])
print("vstack:\n", np.vstack([v1, v2, v2, v1]))

print("*** Horizontal Stacking Vectors ***")
print("hstack:\n", np.hstack([v1, v2, v2, v1]))
print("******************")

print("*** Miscellaneous ***")
ms = np.genfromtxt("data.txt", delimiter=',')
print("Il file data contiene:\n", ms)

# Advanced Boolean Masking Indexing
print("Return true o false if ms > 30", ms > 30)

# Can index with a list in NumPy
kl = np.array([1,2,3,4,5,6,7,8,9])
print("Return element in position (index) 1-2-8:\n", kl[[1, 2, 8]])
