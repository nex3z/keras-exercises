import numpy as np
from keras.datasets import boston_housing


class BostonData(object):
    def __init__(self, normalize=True):
        (train_x, train_y), (test_x, test_y) = boston_housing.load_data()

        if normalize:
            mean = train_x.mean(axis=0)
            train_x -= mean
            std = train_x.std(axis=0)
            train_x /= std
            test_x -= mean
            test_x /= std

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def get_fold(self, fold, k):
        total_x = self.train_x
        total_y = self.train_y
        num_val_samples = len(total_x) // k

        val_idx = np.arange(fold * num_val_samples, (fold + 1) * num_val_samples)
        val_x = total_x[val_idx, ]
        val_y = total_y[val_idx, ]

        train_idx = np.setdiff1d(np.arange(0, len(total_x)), val_idx)
        train_x = total_x[train_idx]
        train_y = total_y[train_idx]

        return (train_x, train_y), (val_x, val_y)
