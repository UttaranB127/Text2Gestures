import numpy as np
import os

filename = './outfile.npy'
B = np.load(filename)
print(B.shape)