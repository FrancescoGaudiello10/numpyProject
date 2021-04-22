# Rif: https://www.youtube.com/watch?v=8Y0qQEh7dJg&list=PLzoetWSUagaa8R9xdaBifysYwgQH0KBTZ&index=3&t=1309s

import numpy as np
import matplotlib as plt
from numpy import random

# NumPy Arrays: Creating Arrays
list_1 = [1, 2, 3, 4, 5]                        # Lista Python
np_arr_1 = np.array(list_1, dtype=np.int8)      # Converto lista in array di numpy
print("Array monodimensionale:\n", np_arr_1)

m_list_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]    # Multi-dimensional array
np_m_array_1 = np.array(m_list_1)
print("Array bi-dimensionale:\n", np_m_array_1)

print("arange(1, 10):\n", np.arange(1, 10))             # array da 1 a 10
print("linspace(0, 5, 5)", np.linspace(0, 5, 5))        # 5 elementi da 0 a 5 --> (0., 1.25, 2.5, 3.75, 5)
print("zeros:\n", np.zeros(5))
print("zeros(3,3):\n", np.zeros((3, 3)))
print("ones(2,2):\n", np.ones((2, 2)))

s = np_m_array_1.size   # Ritorna il numero di elementi all'interno
print("size:", s)

# Array random
random_array = np.random.randint(10, 50, 5)     # Array random interi da 10 a 50, size=5
print("random_array:", random_array)
random_array_bidim = np.random.randint(10, 60, size=(2, 2))
print("random_array_bidim:\n", random_array_bidim)

print("*********************")
print("*** SLICING and INDEXES ***")
print("np_m_array_1:\n", np_m_array_1)
np_m_array_1[0, 0] = 2
np_m_array_1.itemset((0, 1), 1)
print("Change and itemset(0, 1):\n", np_m_array_1)

print("Shape:\n", np_m_array_1.shape)
np.put(np_m_array_1, [0, 3, 6], [10, 10, 10])
print("\nnp_m_array_1 put[10,10,10]:\n", np_m_array_1)
print("\nnp_m_array_1[::-1]:\n", np_m_array_1[::-1])

evens = np_m_array_1[np_m_array_1 % 2 == 0]
print("evens:\n", evens)

print("np_m_array_1[np_m_array_1 > 5]:\n", np_m_array_1[np_m_array_1 > 5])
print("np_m_array_1[(np_m_array_1 > 5) & (np_m_array_1 < 9)]:\n", np_m_array_1[(np_m_array_1 > 5) & (np_m_array_1 < 9)])
print("np_m_array_1[(np_m_array_1 < 5) | (np_m_array_1 == 9)]:\n", np_m_array_1[(np_m_array_1 < 5) | (np_m_array_1 == 10)])

print("np.unique(np_m_array_1):\n", np.unique(np_m_array_1))

print("*********************")
print("*** RESHAPING ARRAYS ***")
print("np_m_array_1.reshape((1, 9)):\n", np_m_array_1.reshape((1, 9)))
print("np.resize(np_m_array_1, (2, 5)):\n", np.resize(np_m_array_1, (2, 5)))

print("\nArray iniziale:\n", np_m_array_1)
print("traspose():\n", np_m_array_1.transpose())
print("swapaxes(0,1):\n", np_m_array_1.swapaxes(0, 1))
print("flatten():\n", np_m_array_1.flatten())
print("flatten().shape:\n", np_m_array_1.flatten().shape)
np_m_array_1.flatten('F')
np_m_array_1.sort(axis=1)     # Riordinamento righe
print("Sort()\n", np_m_array_1)

print("*********************")
print("*** STACKING and SPLITTING ***")
ss_arr_1 = np.random.randint(10, size=(2, 2))
print("ss_arr_1:\n", ss_arr_1)
ss_arr_2 = np.random.randint(10, size=(2, 2))
print("ss_arr_2:\n", ss_arr_2)

vstack_array = np.vstack((ss_arr_1, ss_arr_2))
print("vstack_array:\n", vstack_array)
hstack_array = np.hstack((ss_arr_1, ss_arr_2))
print("hstack_array:\n", hstack_array)

ss_arr_3 = np.delete(ss_arr_1, 1, 0)
ss_arr_4 = np.delete(ss_arr_2, 1, 0)
print("ss_arr_3:\n", ss_arr_3)
print("ss_arr_4:\n", ss_arr_4)

print("column_stack:\n", np.column_stack((ss_arr_3, ss_arr_4)))
print("row_stack:\n", np.row_stack((ss_arr_3, ss_arr_4)))

print("*********************")
print("*** COPYING ***")
cp_arr_1 = np.random.randint(10, size=(2, 2))
cp_arr_2 = cp_arr_1
print("cp_arr_1:\n", cp_arr_1)
print("cp_arr_2:\n", cp_arr_2)
cp_arr_1[0, 0] = 2      # La modifica si riflette anche su cp_arr_2
print("cp_arr_1:\n", cp_arr_1)
print("cp_arr_2:\n", cp_arr_2)

cp_arr_3 = cp_arr_1.view()
print("cp_arr_3:\n", cp_arr_3)

cp_arr_4 = cp_arr_1.copy()
print("cp_arr_4:\n", cp_arr_4)

print("*********************")
print("*** BASIC MATH ***")
