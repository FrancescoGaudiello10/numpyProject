# Views and Copies
import numpy as np

mi_casa = np.array([-54, -6, 2, 54, 7, 87, -3, 6])
su_casa = mi_casa

### Same or Different
print(mi_casa is su_casa)

print("mi_casa ID:", id(mi_casa))
print("su_casa ID:",id(su_casa))

# value equality
print("mi_casa == su_casa:\n", mi_casa == su_casa)
su_casa[4] = 1010
print("su_casa:", su_casa)
print("mi_casa:", mi_casa)
print("*************")

tree_house = np.array([-54, -6, 2, 54, 7, 87, -3, 6])
print("tree_house==su_casa:", tree_house==su_casa)
print("tree_house ID:", id(tree_house))
print(tree_house is mi_casa)
print("*************")

print("*** view: a shallow copy ***")
farm_house = tree_house.view()
farm_house.shape = (2, 4)
print("tree_house:\n", tree_house)  # Same location on memory
print("farm_house:\n", farm_house)
tree_house[3] = -111
print("farm_house:\n", farm_house)
print("*************")

print("*** Deep Copy ***")
dog_house = np.copy(tree_house)
dog_house[0] = -1212
print("dog_house:\n", dog_house)    # Allocate in difference cell of memory (ID diverso)
print("tree_house:\n", tree_house)
print("*************")

print("*** Adding and Removing Elements from NumPy Arrays ***")
a = np.array(np.arange(24)).reshape(2, 3, 4)
print("Array a:\n", a)

b = np.append(a, [5, 6, 7, 8])
print("Array b after append():\n", b)
print("b.shape", b.shape)
# Setting reshape
b = b.reshape((7, 4))
print("b:\n", b)

c = np.array(np.arange(24)).reshape(2, 3, 4) * 10 + 3
print("C:\n", c)
print("**************")
print("a, c, axis=0:\n", np.append(a, c, axis=0))

# Per trovare la nuova dimensione di "c"
sh0 = np.append(a, c, axis=0).shape
print("Shape axis = 0:", sh0)

sh1 = np.append(a, c, axis=1).shape
print("Shape axis = 1:", sh1)

sh2 = np.append(a, c, axis=2).shape
print("Shape axis = 2:", sh2)
print("**************")

print("*** Insert Operation ***")
after_insert_array = np.insert(c, 1, 444, axis=0)   # Ho inserito in index = 1 una matrice di "444"
print("after_insert_array (c, 1, 444, axis=0):\n", after_insert_array)

print("\nnp.insert(c, 1, 444, axis=1):\n", np.insert(c, 1, 444, axis=1))
print("\nnp.insert(c, 1, 444, axis=2):\n", np.insert(c, 1, 444, axis=2))

d = np.empty(c.shape)
np.copyto(d, c)
print("copyto_d:\n", d)

print("delete:\n", np.delete(d, 1, axis=0))     # Cancello la matrice con index 1
