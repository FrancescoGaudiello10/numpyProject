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

