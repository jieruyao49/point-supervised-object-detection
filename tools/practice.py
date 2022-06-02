import numpy as np

a = np.zeros((1,4))
a[0,3] = 1
b = len(np.where(a[0] != 0)[0])


print(b.shape)