import matplotlib.pyplot as plt
import numpy as np
np.random.seed(12122)
import keras
import keras.backend as K

 
x = np.random.uniform(-10, 10, 50)
print(x)

x_new = K.reshape(x,(-1,5,2))
print(x_new.shape.eval())
print(x_new.eval())

y = x_new[:,2:4].eval()
print(y)
