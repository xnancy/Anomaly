import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from csv_utils import decode_csv

pie = decode_csv('./pie_pca.csv')
imagen = decode_csv('./imagen_pca.csv')

gmm = GaussianMixture(n_components=2, weights_init = [])

cont = np.vstack((pie, imagen))
gmm.fit(cont)

