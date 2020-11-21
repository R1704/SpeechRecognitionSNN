ttfs_spikes_test.p and ttfs_spikes_train.p are the data that we need to feed to the convnet.
They are stored as numpy arrays by pickle, load them with pickle.load

They both have the same format: [n_samples, n_rows, n_frequency_bins]

In the current version, the values are:
n_rows = 41
n_frequency_bins = 40

The rows are the rows in the spectrogram, the frequency_bins are the frequency bands. 
The values in the array are categorical: use a one-hot encoding to get the spikes, i.e. [0,2,1] becomes     [[1,0,0], 
                                                                                                            [0,0,1],
                                                                                                            [0,1,0]]
                                                                                                        