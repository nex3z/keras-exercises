import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

_DOWNLOAD_URL = 'https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip'
_DOWNLOAD_FILE_NAME = os.path.basename(_DOWNLOAD_URL)
_FILE_NAME = _DOWNLOAD_FILE_NAME[:-4]


class JenaClimate:
    def __init__(self, directory='./data', num_train=200000, num_val=100000, look_back=1440, step=6, delay=144,
                 batch_size=128):
        file_path = _maybe_download(directory)
        self.df_data = pd.read_csv(file_path)
        length = len(self.df_data)

        self.train_from_idx = 0
        self.train_to_idx = self.train_from_idx + num_train
        self.val_from_idx = self.train_to_idx
        self.val_to_idx = self.train_to_idx + num_val
        self.test_from_idx = self.val_to_idx
        self.test_to_idx = length

        self.look_back = look_back
        self.step = step
        self.delay = delay
        self.batch_size = batch_size

        self.val_steps = (num_val - look_back) // batch_size
        num_test = length - num_train - num_val
        self.test_steps = (num_test - look_back) // batch_size

        np_data = self.df_data.drop(columns='Date Time').to_numpy()
        self.mean = np_data[self.train_from_idx:self.train_to_idx].mean(axis=0)
        np_data -= self.mean
        self.std = np_data[self.train_from_idx:self.train_to_idx].std(axis=0)
        np_data /= self.std
        self.np_data = np_data

    def build_generator(self):
        train_gen = self.generator(self.train_from_idx, self.train_to_idx, shuffle=True)
        val_gen = self.generator(self.val_from_idx, self.val_to_idx)
        test_gen = self.generator(self.test_from_idx, self.test_to_idx)
        return train_gen, val_gen, test_gen

    def generator(self, idx_from, idx_to, shuffle=False):
        idx = idx_from + self.look_back

        while True:
            if shuffle:
                batch_idx_to = np.random.randint(idx_from + self.look_back, idx_to, size=self.batch_size)
            else:
                if idx + self.batch_size >= idx_to:
                    idx = idx_from + self.look_back
                batch_idx_to = np.arange(idx, min(idx + self.batch_size, idx_to))
                idx += len(batch_idx_to)

            samples = np.zeros((len(batch_idx_to), self.look_back // self.step, self.np_data.shape[-1]))
            targets = np.zeros((len(batch_idx_to),))

            for i, sample_idx_to in enumerate(batch_idx_to):
                sample_indices = range(sample_idx_to - self.look_back, sample_idx_to, self.step)
                target_index = sample_idx_to + self.delay
                samples[i] = self.np_data[sample_indices]
                targets[i] = self.np_data[target_index][1]
            yield samples, targets


def _maybe_download(directory):
    file_path = os.path.join(directory, _FILE_NAME)

    if not os.path.isfile(file_path):
        if not os.path.isdir(directory):
            os.mkdir(directory)

        download_path = os.path.join(directory, _DOWNLOAD_FILE_NAME)
        if not os.path.isfile(download_path):
            urllib.request.urlretrieve(_DOWNLOAD_URL, download_path)

        zip_file = zipfile.ZipFile(download_path, 'r')
        zip_file.extractall(directory)
        zip_file.close()

    return file_path
