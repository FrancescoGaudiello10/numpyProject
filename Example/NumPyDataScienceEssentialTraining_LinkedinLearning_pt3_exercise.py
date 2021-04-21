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
