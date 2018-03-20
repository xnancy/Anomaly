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

N = 200
shaper = csr_matrix((N, ncols))
srp = SparseRandomProjection(eps = 0.5)
# srp.fit(shaper)

print("Loading pie vectors...")
pie = decode_csv("./pie_prepool.csv")
print(pie.shape)
print("Loading sushi vectors...")
imagen = decode_csv("./sushi_prepool.csv")
print(imagen.shape)


print("Merging arrays...")
contaminated = np.vstack((pie, imagen))

#print("Sparse random projection...")
#small = srp.fit_transform(contaminated)

print("Computing principal components...")
n_c = 2000

spca = MiniBatchSparsePCA(n_components=n_c, verbose = 1)
spca.fit(contaminated)

spca_pie = spca.transform(pie)
spca_imagen = spca.transform(imagen)

csv_utils.encode_csv('./spca_pie.csv', spca_pie)
csv_utils.encode_csv('./spca_sushi.csv', spca_imagen)

pca = PCA(n_components = n_c)
pca.fit(contaminated)

pca_pie = pca.transform(pie)
pca_imagen = pca.transform(imagen)

csv_utils.encode_csv('./pca_pie.csv', pca_pie)
csv_utils.encode_csv('./pca_sushi.csv', pca_imagen)

