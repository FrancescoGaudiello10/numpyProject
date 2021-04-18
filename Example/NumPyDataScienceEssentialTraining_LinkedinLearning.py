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
my_vector = np.array([-12, -4, 6, 5, 10, 32, 0, -5])
print("My vector:\n", my_vector)
print("my_vector[0]: ", my_vector[0])
print("Change first element...")
my_vector[0] = -50
print("New my_vector:\n", my_vector)

print("*** Two Dimensional Arrays ***")
my_array = np.arange(35)    # Ho 35 elementi
my_array.shape = (7, 5)     # Divisi in 7 righe e 5 colonne
print("My_array:\n", my_array)

# Access row or column
print("my_array[2]: ", my_array[2])
print("my_array[5, 2]: ", my_array[5, 2])

print("\nAnother case to represent an element is")
print("my_array[5][2]: ", my_array[5][2])
print("***************")

print("*** Three dimensional arrays ***")
my_3D_array = np.arange(70)
my_3D_array.shape = (2, 7, 5)   # 2 matrici di ognuna 7 righe e 5 colonne
print("Il mio array 3D:\n", my_3D_array)

print("\nmy_3D_array[1] -> take a matrix with index 1 (the second in this case):\n", my_3D_array[1])
print("\nmy_3D_array[1, 3] -> take a matrix with index 1 and row 3 index with all column:\n", my_3D_array[1, 3])
print("\nmy_3D_array[1, 3, 2] -> take a matrix with index 1 and row 3 index and column 2:\n", my_3D_array[1, 3, 2])
print("Change element in [1, 3, 2]...")
my_3D_array[1, 3, 2] = 1111
print(my_3D_array)

# Boolean Mask Array
n = np.array([-7, -12, 4, 65, -76, 63, -6, 20, -5])
print("Gli elementi dell'arrays ono divisibili per 7:")
zero_mod_7_mask = 0 == (n % 7)
print(zero_mod_7_mask)
sub_array = n[zero_mod_7_mask]
print("Il sub_array booleano:", sub_array)
print("***************")

# Numpy Logicla Operators
mod_tes = 0 == (n % 7)
positive_test = n > 0
print("Positive_Test (solo positivi) = ", positive_test)

combined_mask = np.logical_and(mod_tes, positive_test)
print("Combined_mask (AND):", combined_mask)
print(n[combined_mask])
combined_mask_or = np.logical_or(mod_tes, positive_test)
print("Combined_mask_or (OR):", combined_mask_or)
print(n[combined_mask_or])
print("***************")

print("*** Broadcasting ***")
x = np.arange(70)
x.shape = (2, 7, 5)
print("my_3d_array:\n", x)
print("x.shape: ", x.shape)
print("x.ndim: ", x.ndim)
print("x.size: ", x.size)
print("x.dtype: ", x.dtype)

print(5 * x - 2)
print("***************")

left_mat = np.arange(6).reshape((2, 3))
right_mat = np.arange(15).reshape((3, 5))
print("Dot:", np.dot(left_mat, right_mat))

print("*** Operations along axes ***")
print("x.shape: ", x.shape)
print("x.sum(): ", x.sum())
print("x.sum(axis=0):\n", x.sum(axis=0))     # axis=0 = colonne
print("x.sum(axis=1):\n", x.sum(axis=1))     # axis=1 = righe
print("***************")

print("*** Creaing Structured Arrays ***")
person_data_def = [('name', 'S5'), ('height', 'f8'), ('weight', 'f8'), ('age', 'i8')]
print("Person_data_def:", person_data_def)

people_array = np.zeros(4, dtype=person_data_def)
print("People_array:", people_array)
people_array[3] = ('Delta', 73, 205, 34)
print("People_array:", people_array)
people_array[0] = ('Alfa', 65, 112, 23)
print("People_array:", people_array)
ages = people_array['age']
print("Ritorna 'age' di tutte le persone sotto forma di array: ", ages)
print("***************")

print("*** Multi-dimensional Structured Arrays ***")
people_big_array = np.zeros((4, 3, 2), dtype=person_data_def)
print("people_big_array:\n", people_big_array)
people_big_array[3, 2, 1] = ('Echo', 68, 155, 46)
print("***************")
print("people_big_array:\n", people_big_array)
print("***************")

print("people_big_array['height']:\n", people_big_array['height'])
print("***************")

print("*** Creating Record Array ***")
person_record_array = np.rec.array([('Delta', 54, 65, 12), ('John', 96, 143, 9)], dtype=person_data_def)
print("person_record_array:\n", person_record_array)