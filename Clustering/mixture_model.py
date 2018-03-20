import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from csv_utils import decode_csv

pie = decode_csv('./pca_pie.csv')
imagen = decode_csv('./pca_sushi.csv')

gmm = GaussianMixture(n_components=2, max_iter = 10000)

cont = np.vstack((pie, imagen))
gmm.fit(cont)
yhat = gmm.predict(cont)

z1 = np.mean(yhat[:1000])
z2 = np.mean(yhat[1000:])
print(z1, z2)

plt.plot(pie[:, 0], pie[:, 1], 'r+', imagen[:, 0], imagen[:, 1], 'bo')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.savefig('./plot.png')
