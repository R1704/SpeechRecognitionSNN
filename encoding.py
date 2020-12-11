from scipy.io import loadmat
import pandas as pd

data = loadmat('data/Spike-TIDIGITS.mat')

train_labels = data['train_labels']
test_labels = data['test_labels']
train_pattern = data['train_pattern']
test_pattern = data['test_pattern']
print(train_labels.shape, train_pattern.shape)
