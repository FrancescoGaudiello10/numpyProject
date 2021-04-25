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

print("*** STACKING TOGETHER DIFFERENT ***")
print("m:\n", m)
print("n:\n", n)
print("********")
mn_vstack = np.vstack((m, n))
print(mn_vstack)
print("********")
mn_hstack = np.hstack((m, n))
print(mn_hstack)
print("********")

# columns_stack inserisce in sotto una colonna unificata (valido solo per 1D)
mn_columnstack = np.column_stack((m, n))
print(mn_columnstack)
a = np.array([1, 2])
b = np.array([5, 7])
print("********")
ab_columnstack = np.column_stack((a, b))
print(ab_columnstack)

print("*** NO COPY AT ALL ***")
a = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 12]])
b = a                       # no new object is created
print("b is a:", b is a)    # a and b are two names for the same ndarray object


def f(x):
    print(id(x))


# Posso notare che puntano allo stesso indirizzo di mem.
print("id(a):", id(a))
print("id(b):", id(b))

# View or shallow copy
# view() crea un nuovo array object con gli stessi data
c = a.view()
print("c is a:", c is a)
print("id(a):", id(a))
print("id(c):", id(c))
c = c.reshape((2, 6))
print(a.shape)
print(c.shape)

# Noto che modificando un data di "c"
# viene modificato anche "a"
c[0, 4] = 12345
print(a)

# DEEP COPY
# The copy method makes a complete copy of the array and its data.
d = a.copy()
print(d is a)
print(d.base is a)
print("d:\n", d)
print("a:\n", a)
print("***** modified d[0,0]...")
d[0, 0] = 9999      # d doesn't share anything with a
print("d:\n", d)
print("a:\n", a)

# example utility copy
a = np.arange(int(1e8))
b = a[:100].copy()
del a                       # the memory of ``a`` can be released

print("*** Functions and Methods Overview ***")
"""
List of some useful Numpy function: 
Array Creation:
    arange, array, copy, empty, empty_like, eye, fromfile, 
    fromfunction, identity, linspace, logspace, mgrid, ogrid, 
    ones, ones_like, r_, zeros, zeros_like

Conversions:
    ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat

Manipulations:
    array_split, column_stack, concatenate, diagonal, dsplit, 
    dstack, hsplit, hstack, ndarray.item, newaxis, ravel, 
    repeat, reshape, resize, squeeze, swapaxes, take, 
    transpose, vsplit, vstack

Questions:
    all, any, nonzero, where

Ordering:
    argmax, argmin, argsort, max, min, ptp, searchsorted, sort

Operations:
    choose, compress, cumprod, cumsum, inner, 
    ndarray.fill, imag, prod, put, putmask, real, sum

Basic Statistics:
    cov, mean, std, var

Basic Linear Algebra:
    cross, dot, outer, linalg.svd, vdot
"""

print("*** LESS BASIC ***")
print("Bradcasting rules")
# Broadcasting allows universal functions to deal in a meaningful
# way with inputs that do not have exactly the same shape.
# The first rule of broadcasting is that if all input arrays
# do not have the same number of dimensions, a “1” will be
# repeatedly prepended to the shapes of the smaller arrays until
# all the arrays have the same number of dimensions.
#
# The second rule of broadcasting ensures that arrays
# with a size of 1 along a particular dimension act as
# if they had the size of the array with the largest shape along that dimension.
# The value of the array element is assumed to be the same along that
# dimension for the “broadcast” array.
# Rif: https://numpy.org/doc/stable/user/basics.broadcasting.html#basics-broadcasting

print("*** Advanced indexing and index tricks ***")
a = np.arange(12)**2
print("a:\n", a)
i = np.array([1, 1, 3, 8, 5])
print("i:\n", i)
print("a[i]:", a[i])        # Stampa gli elementi secondo l'indice "i" presenti in "a"

j = np.array([[3, 4],
              [9, 7]])
print("j:\n", j)
print("a[j]:\n", a[j])      # Eleva al quadrato gli elementi di j
print("************")

print("*** PALETTE ***")
palette = np.array([[0, 0, 0],
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [255, 255, 255]])
print("palette:\n", palette)
image = np.array([[0, 1, 2, 0],
                  [0, 3, 4, 0]])
print("image:\n", image)
print("palette[image]:\n", palette[image])

a = np.arange(12).reshape(3, 4)
print("a:\n", a)

i = np.array([[0, 1],
              [1, 2]])
j = np.array([[2, 1],
              [3, 3]])
print("i:\n", i)
print("j:\n", j)
print("a[i, j]:\n", a[i, j])
print("********")

a = np.arange(12).reshape(3, 4)
print("a:\n", a)
i = np.array([[0, 1],
              [1, 2]])
j = np.array([[2, 1],
              [3, 3]])
print("i:\n", i)
print("j:\n", j)

print("a[i, j]:\n", a[i, j])
print("*********")

print("*** Indexing with Boolean Arrays ***")
a = np.arange(12).reshape(3, 4)
b = a > 4
print(b)        # Array Booleano di true/false
print(a[b])     # Stampa i soli valori TRUE
a[b] = 0        # All elements of 'a' higher than 4 become 0
print(a)

print("*********")
print("*** Mandelbrot set ***")
import matplotlib.pyplot as plt
def mandelbrot(h, w, maxit=20):
    """ RETURNS AN IMAGE OF THE MANDELBROT FRACTAL OF SIZE (H,W) """
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == maxit)
        divtime[div_now] = i
        z[diverge] = 2
    return divtime

#plt.imshow(mandelbrot(400, 400))

# second way to index with boolena
a = np.arange(12).reshape(3, 4)
b1 = np.array([[False, True, True]])
b2 = np.array([True, False, True, False])

# print("a[b1, :]:\n", [a[b1, :]])
print("************")
print("*** TRICKS AND TIPS ***")
print("Automatic Reshaping")
