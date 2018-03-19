import numpy as np

def encode_csv(filename, X):
    np.savetxt(filename, X, delimiter=',')

def decode_csv(filename):
    return np.loadtxt(filename, delimiter=',')    
