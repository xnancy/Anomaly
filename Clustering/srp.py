import csv_utils

import numpy as np
import pandas as pd
import csv
import gc
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.random_projection import SparseRandomProjection

ncols = 98305

def decode_csv(filename):
    names = [str(i) for i in range(ncols)]
    data = pd.read_csv(filename, header=None, names=names)
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

# srp.fit(shaper)

print("Loading pie vectors...")
pie = decode_csv("./pie_prepool.csv")
print(pie.shape)
print("Loading sushi vectors...")
imagen = decode_csv("./imagen_prepool_rerun.csv")
print(imagen.shape)


print("Merging arrays...")
contaminated = np.vstack((pie, imagen))

print("Sparse random projection...")

# print("Computing principal components...")
# n_c = 500

# spca = MiniBatchSparsePCA(n_components=n_c, verbose = 1)
# spca.fit(contaminated)
srp = SparseRandomProjection()
small = srp.fit(contaminated)
srp_pie = srp.transform(pie)
srp_imagen = srp.transform(imagen)
print(srp_pie.shape)
csv_utils.encode_csv('./srp_pie2_large.csv', srp_pie)
csv_utils.encode_csv('./srp_imagen_large.csv', srp_imagen)
