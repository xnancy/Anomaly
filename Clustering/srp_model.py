import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from csv_utils import decode_csv

pie = decode_csv('./srp_pie2_large.csv')
imagen = decode_csv('./srp_imagen_large.csv')
# pie = decode_csv('./pie_prepool.csv')[1:]
# imagen = decode_csv('./sushi_prepool.csv')[1:]

# Dim = 2000
# pie = pie[:, Dim]
# imagen = imagen[:, Dim]

def pct(x):
    return 1000 * round(x, 3)

def solve(dim, proportion = 1.0):
    print(proportion)
    num_other = int(1000 * proportion)
    cont = np.vstack((pie, imagen[:num_other]))[:, :dim]
    mean_0 = np.mean(cont[:1000], axis = 0)
    mean_1 = np.mean(cont[1000:], axis = 0)
    gmm = GaussianMixture(n_components=2, max_iter = 10000, means_init = [mean_0, mean_1], weights_init=[1 / (1 + proportion), proportion / (1 + proportion)])
    gmm.fit(cont)
    yhat = gmm.predict(cont)

    est1 = yhat[:1000]
    est2 = yhat[1000:]

    z1 = np.mean(yhat[:1000])
    z2 = np.mean(yhat[1000:])   
 
    print(z1, z2)
    yhat = 1 - yhat
    if z2 < 0.5:
        z1, z2 = 1 - z1, 1 - z2
        yhat = 1 - yhat

    red = cont[yhat == 0]
    blue = cont[yhat == 1]

    red_pie = cont[:1000][est1 == 0]
    red_other = cont[1000:][est2 == 0]
    blue_pie = cont[:1000][est1 == 1]
    blue_other = cont[1000:][est2 == 1]
    
    # plt.axes('3d')
    plt.plot(red_pie[:, 0], red_pie[:, 1] ,'r+', 
        red_other[:, 0], red_other[:, 1], 'ro',
        blue_pie[:, 0], blue_pie[:, 1], 'b+', 
        blue_other[:, 0], blue_other[:, 1], 'bo'
    )
    plt.title('theme-pie v. anomaly', fontweight="bold")
    plt.xlabel('SRP1', fontweight="bold")
    plt.ylabel('SRP2', fontweight="bold")
    plt.savefig('./plot' + str(num_other) + '_' + str(dim) + '_p'+ str(pct(z1)) + '_r'+str(pct(1 - z1)) +'srp.png')
    plt.clf()
for dim in [5, 10, 50, 100, 250, 500, 6000]:
    solve(dim, .333)
