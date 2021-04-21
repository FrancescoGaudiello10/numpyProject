# Universal Function
# Rif: https://www.linkedin.com/learning/numpy-data-science-essential-training/universal-functions

import numpy as np
# truncated binomial: return (x+1)**3 - (x)**3
print("*** Universal Function ***")
def truncated_binomial(x):
    return (x + 1) ** 3 - x ** 3

# Use to function testing
# assert_equal = (funzione che richiami, risultato che ti attendi)
print("assert_equal:", np.testing.assert_equal(truncated_binomial(4), 61))
my_numpy_function = np.frompyfunc(truncated_binomial, 1, 1)
print("frompyfunc:", my_numpy_function)

test_array = np.arange(10)
print(my_numpy_function(test_array))

big_test_array = np.outer(test_array, test_array)
print("big_test_array:\n", big_test_array)
print(my_numpy_function(big_test_array))
print("********************")

print("*** Pythagorean Triangles ***")
def is_integer(x):
    return np.equal(np.mod(x, 1), 0)

numpy_is_integer = np.frompyfunc(is_integer, 1, 1)
print("numpy_is_integer:\n", numpy_is_integer)

number_of_triangles = 9
base = np.arange(number_of_triangles) + 1
height = np.arange(number_of_triangles) + 1

hypotenuse_squared = np.add.outer(base **2 , height ** 2)
hypotenuse = np.sqrt(hypotenuse_squared)
print("hypotenuse:", hypotenuse)
print("numpy_is_integer(hypotenuse):\n", numpy_is_integer(hypotenuse))
print("************************")


print("*** LINEAR ALGEBRA ***")
my_first_matrix = np.matrix([[3, 1, 4], [5, 7, 9], [8, 4, 7]])
print("my_first_matrix:\n", my_first_matrix)

print("my_first_matrix.T - Trasposta:\n", my_first_matrix.T)
print("my_first_matrix.I - Inversa:\n", my_first_matrix.I)

my_first_matrix * my_first_matrix.I

# To create a matrix identity:
print("Identity 5 \n", np.eye(5))

# Solve simultaneous linear equations
right_hand_side = np.matrix([[11], [22], [33]])
my_first_inverse = my_first_matrix.I
solution = my_first_inverse * right_hand_side
print("Solution:\n", solution)
print("***********")

# More efficient solution for large matrix
from numpy.linalg import solve
print("SOLVE:")
print(solve(my_first_matrix, right_hand_side))
print("************************")

print("*** BRAIN TEASERS ***")
np.set_printoptions(precision=2)

# Version of Numpy
print(np.__version__)

# How reverse an array
my_array = np.arange(10)
my_array_reverse = my_array[:: -1]

# Triplica il valore di ogni elemento creato precedentemente
my_triple_array = my_array * 3

# Create array with 20 zero; every fifth element equals four
my_zero_array = np.zeros(20)
my_zero_array[0::5] = 4

# Create a 5x5 identity matrix with integer components
print(np.identity(5))
# Oppure
my_matrix = np.asmatrix(np.eye(5, dtype='int'))
print(my_matrix)

# Find the mean of a vector with 30 random elements
print("*** Mean ***")
mean_random = np.random.random(30)
print(mean_random.mean())

# Create an 8x8 checker board with alternating zeros and oens
my_checker_board = np.zeros((8, 8), dtype=int)
my_checker_board[1::2, ::2] = 1
my_checker_board[::2, 1::2] = 1
print(my_checker_board)

# Create a sorted vecotr that contain "n" random numbers
vector = 12
my_random_vector = np.random.random(vector)
my_sorted_vector = np.sort(my_random_vector)
print("Vettore ordinato:\n", my_sorted_vector)

# Withou sorting, replace largest element in random array with the value 12
print("*** Random vector ***")
my_random_vector[my_random_vector.argmax()] = 12
print(my_random_vector)

# Given the followin data type and data set, sort by height
camelot_dtype = [('name', 'S10'), ('height', float), ('age', int)]
camelot_values = [('Arthur', 1.8, 41), ('Lancilo', 1.7, 23), ('Mark', 2.1, 33)]

camelot_structured_array = np.array(camelot_values, dtype=camelot_dtype)
camelot_sorted_array = np.sort(camelot_structured_array, order='height')
print(camelot_sorted_array)
for n in np.arange(camelot_sorted_array.size):
    print(camelot_sorted_array[n])

# Make an array read_only (immutable)
my_ordinary_array = np.array(np.arange(12))
my_ordinary_array.flags.writeable = False
# my_ordinary_array[5] = 4

# Print enumerated values form a 3z3 Numpy array
my_3_3_array = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(my_3_3_array):
    print(index, value)