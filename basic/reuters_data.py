from keras.datasets import reuters
from keras.utils import to_categorical

from common.DataSet import DataSet, Datasets
from common.util import to_vector


def read_reuters_data(num_words=10000, validation_size=1000):
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)

    train_data = to_vector(train_data, num_words)
    train_labels = to_categorical(train_labels)

    validation = DataSet(data=train_data[:validation_size], labels=train_labels[:validation_size])
    train = DataSet(data=train_data[validation_size:], labels=train_labels[validation_size:])

    test_data = to_vector(test_data, num_words)
    test_labels = to_categorical(test_labels)

    test = DataSet(data=test_data, labels=test_labels)

    return Datasets(train=train, validation=validation, test=test)
