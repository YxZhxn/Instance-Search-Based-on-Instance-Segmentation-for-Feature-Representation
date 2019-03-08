import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


if __name__ == '__main__':

    input = '/home/yz/cde/fcis/output/feature/pca/ftr-res34-fcisx.txt'
    output = '/home/yz/cde/fcis/output/feature/pca/pca-res34-fcisx.pkl'

    feature = np.loadtxt(input)
    print feature.shape
    dim = feature.shape[1]
    pca = PCA(dim, whiten=True)
    pca.fit(feature)
    pickle.dump(pca, open(output, 'wb'))