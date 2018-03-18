import numpy as np
import pandas as pd
import csv
import gc
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection

ncols = 98305

def decode_csv(filename):
    names = [str(i) for i in range(ncols)]
    data = pd.read_csv('./pie_prepool.csv', header=None, names=names)
    del data['0']
    val = data.values
    del data
    gc.collect()
    return val

def decode_lines_of_csv(filename, lines):
    ret = np.zeros((0, ncols))
    with open(filename, 'r') as csvfile:
        i = 0
        reader = csv.reader(csvfile)
        for row in reader:
            if i % 50 == 0:
                print(i)
            if i == lines:
                break
            i += 1
            ret = np.vstack((ret, row))
    return ret[:, 1:]

N = 200
shaper = csr_matrix((N, ncols))
srp = SparseRandomProjection(eps = 0.5)
# srp.fit(shaper)

print("Loading pie vectors...")
# pie = decode_csv("./pie_prepool.csv")
# pie = pie[1:190]
pie = decode_lines_of_csv("./pie_prepool.csv", 190)
print("Loading imagen vectors...")
# imagen = decode_csv("./imagen_prepool_rerun.csv")
# imagen = imagen[1:10]
imagen = decode_lines_of_csv("./imagen_prepool_rerun.csv", 10)
print("Merging arrays...")
contaminated = np.vstack((pie, imagen))
del pie
del imagen

print("Sparse random projection...")
small = srp.fit_transform(contaminated)

del contaminated


