import numpy as np
from keras.datasets import imdb

from common.DataSet import DataSet, Datasets
from common.util import to_vector


def read_imdb_data(num_words=10000, validation_size=10000):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

    train_data = to_vector(train_data, num_words)
    train_labels = np.asarray(train_labels).astype('float32')

    validation = DataSet(data=train_data[:validation_size], labels=train_labels[:validation_size])
    train = DataSet(data=train_data[validation_size:], labels=train_labels[validation_size:])

    test_data = to_vector(test_data, num_words)
    test_labels = np.asarray(test_labels).astype('float32')

    test = DataSet(data=test_data, labels=test_labels)

    return Datasets(train=train, validation=validation, test=test)
