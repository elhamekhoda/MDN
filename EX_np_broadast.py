import numpy as np

# Initialize `x` and `y`
x = np.ones((3,4))
y = np.random.random((5,1,4))
print("printing x \n", x)
print("printing y \n", y)

# Add `x` and `y`
print(x + y)