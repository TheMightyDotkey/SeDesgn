

print('test2')

import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

normaldata1 = sp.loadmat('normalset1.mat')
normaldataset1 = normaldata1['fftdatanormalset1']
normaldata2 = sp.loadmat('normalset2.mat')
normaldataset2 = normaldata2['fftdatanormalset2']
normaldata3 = sp.loadmat('normalset3.mat')
normaldataset3 = normaldata3['fftdatanormalset3']
innerrace1 = sp.loadmat('12kDEinnerrace7dia0hpset1.mat')
innerraceset1 = innerrace1['fftdata12kDEinnerrace7dia0hpset1']
innerrace2 = sp.loadmat('12kDEinnerrace7dia0hpset2.mat')
innerraceset2 = innerrace2['fftdata12kDEinnerrace7dia0hpset2']
innerrace3 = sp.loadmat('12kDEinnerrace7dia0hpset3.mat')
innerraceset3 = innerrace3['fftdata12kDEinnerrace7dia0hpset3']
in3 = np.reshape(innerraceset3, (1,-1))



testset = [normaldataset1, normaldataset2, normaldataset3, innerraceset1, innerraceset2]
flags = [0, 0, 0, 1, 1]

clf = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes=(5,2), random_state=1)

clf.fit(testset, flags)

clf.predict(innerraceset3)

print('ok')