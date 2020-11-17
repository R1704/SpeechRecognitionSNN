import pandas as pd
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
data = loadmat("data/TIDIGIT_test.mat")

plt.plot(data['test_samples'][2][0])
plt.show()
