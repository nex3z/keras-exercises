import collections


class DataSet(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
