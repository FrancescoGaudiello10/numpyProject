# Rif: https://github.com/rougier/numpy-100

# 1. Import the numpy package under the name np
from datetime import time
from time import time_ns

import numpy as np

# 2. Print the numpy versione and the configuration
print("Versione:", np.__version__)
np.show_config()
print("********************")

# 3. Create a null vector of size 10
z = np.zeros(10)
print("Z:\n", z)
print("********************")

# 4. How to find the memory size of any array
# int = 8 byte
# 10*10*8 = 800 bytes
z = np.zeros((10, 10))
print(("%d bytes" %(z.size * z.itemsize)))
print("********************")

# 5. How to get the documentation of the numpy add function from the command line
# %run `python -c "import numpy; numpy.info(numpy.add)"`

# 6. Create a null vector of size 10 but the fifth value which is 1
z = np.zeros(10)
z[4] = 1
print("Z:\n", z)
print("********************")

# 7. Create a vector with values ranging from 10 to 49
z = np.arange(10, 50)
print("Z:\n", z)
print("********************")

# 8. Reverse a vector (first element becomes last)
z = np.array([1, 2, 3, 4, 5, 6, 7])
z = z[::-1]         # Reverse
print("Z:\n", z)
print("********************")

# 9. Create a 3z3 matrix with values ranging from 0 to 8
z = np.arange(9).reshape((3, 3))
print("Z:\n", z)
print("********************")


# 10. Find indices of non_zero elements from [1, 2, 0, 0, 4, 0]
z = np.array([1, 2, 0, 0, 4, 0])
z = np.where(z != 0)
print("Z:\n", z)
z2 = np.nonzero([1, 2, 0, 0, 4, 0])
print("Z2:\n", z2)
print("********************")

# 11. Create a 3z3 identity matrix
z1 = np.identity(3)
print("Identity:\n", z1)
z2 = np.eye(3)
print("eye:\n", z2)
print("********************")

# 12. Create a 3x3x3 array with random values
z1 = np.random.random((3, 3, 3))
print("Random:\n", z1)
print("********************")

# 13. Create a 10x10 array with random values and findthe minimum and maximum values
z = np.random.randint(0, 500, size=(10, 10))
Zmin, Zmax = z.min(), z.max()
print("Random Integer:\n", z)
print("Min:", Zmin, "\nMax:", Zmax)
print("********************")

# 14. Create a random vector of size 30 and find the mean value
z = np.random.randint(0, 50, size=30)
zmean = z.mean()
print("Mean:\n", zmean)
print("********************")

# 15. Create a 2d array with 1 on the border and 0 inside
z = np.ones((10, 10))
z[1:-1, 1:-1] = 0
print(z)
print("********************")

# 16. How to add a border (filled with 0's) around an existing array?
z = np.ones((5, 5))
z = np.pad(z, pad_width=1, mode='constant', constant_values=0)
print(z)

# Using fancy indexing
z[:, [0, 1]] = 0
z[[0, 1], :] = 0
print(z)
print("********************")

# 17. What is the result of the following expression?
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)
print("********************")

# 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
z = np.diag(1+np.arange(4), k=-1)
print("Diag:\n", z)

# 19. Create a 8x8 matrix and fill it with a checkerboard pattern
z = np.zeros((8, 8),dtype=int)
z[1::2, ::2] = 1
z[::2, 1::2] = 1
print("Z:\n", z)
print("********************")

# 20. Consider a (6, 7, 8) shape array, what is the index (x, y, z) of the 100th elements?
print(np.unravel_index(99, (6, 7, 8)))