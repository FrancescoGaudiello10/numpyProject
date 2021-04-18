# Create NumPy arrays using Python's array like data types
import numpy as np

print("La versione di numpy:", np.__version__)

# Array in Python
my_list = [-12, 6, 7, 9, -3]

# Array con NumPy
my_narray_list = np.array(my_list)
print("My array:\n", my_narray_list)
print("Multiplied * 10:\n", my_narray_list*10)

# Intrinsic NumPy array creation
# Arange = crea un array da 0 fino al numero passato in input
# Rif: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
ar = np.arange(5)
print("Arange:\n", ar)

# Arange da 10 fino a 30 -> [10, 30)
ar2 = np.arange(10, 30)
print("Second arange (10, 30):\n", ar2)

# Length of Array
print("len(a) -> ", len(ar2))
print("a.size -> ", ar2.size)

# arange steps
step = np.arange(10, 30, 2)
print("Arange with step (2): ", step)
print("*********************")

print("*** Linspace - Zeros - Ones ***")
# Rif: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html?highlight=linspace#numpy.linspace
a = np.linspace(2, 3, 5)
b = np.linspace(2, 3, 10)
print("- linspace(2,3,5):\n", a)
print("- linspace(2,3,10):\n", b)

# Zeros
z = np.zeros(5)
print("- zeros(5) -> ", z)
o = np.ones(5)
print("- ones(5) -> ", o)

# n-dimensional arrays
print("N-Dimensional Array\n")
print(np.zeros((5, 4)))     # 5 rows - 4 columns
print("***************")
print(np.ones((5, 4)))
print("***************")

# With 3 elements
multizeros = np.zeros((5, 4, 3))     # 5 matrix with every 4 rows and 3 columns
print(multizeros)
print("***************")

print("*** Index, Slice and Iterate ***")
