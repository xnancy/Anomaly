import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from csv_utils import decode_csv

pie = decode_csv('./pca_pie.csv')
imagen = decode_csv('./pca_sushi.csv')

def solve(proportion = 1.0):
    print(proportion)
    num_other = int(1000 * proportion)
    gmm = GaussianMixture(n_components=2, max_iter = 10000, weights_init=[1 / (1 + proportion), proportion / (1 + proportion)])

    cont = np.vstack((pie, imagen[:num_other]))
    gmm.fit(cont)
    yhat = gmm.predict(cont)
    
    est1 = yhat[:1000]
    est2 = yhat[1000:]

    z1 = np.mean(yhat[:1000])
    z2 = np.mean(yhat[1000:])
    print(z1, z2)

    red = cont[yhat == 0]
    blue = cont[yhat == 1]

    red_pie = red[:1000]
    red_other = red[1000:]
    blue_pie = blue[:1000]
    blue_other = blue[1000:]

    plt.plot(red_pie[:, 0], red_pie[:, 1], 'r+', red_other[:, 0], red_other[:, 1], 'ro',
        blue_pie[:, 0], blue_pie[:, 1], 'b+', blue_other[:, 0], blue_other[:, 1], 'bo')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.savefig('./plot_' + str(num_other) + '.png')

for i in range(1, 10):
    other = i / 10.0
    solve(other)
