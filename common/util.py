import matplotlib.pyplot as plt
import numpy as np


def to_vector(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def moving_average(s, factor=0.9):
    ret = np.empty(len(s))
    for i in range(len(s)):
        if i == 0:
            ret[i] = s[i]
        else:
            ret[i] = factor * ret[i - 1] + (1 - factor) * s[i]
    return ret


def plot_history(train, validation, ylabel=''):
    epochs = range(1, len(train) + 1)
    plt.figure()
    plt.plot(epochs, train, marker='o', label='training')
    plt.plot(epochs, validation, marker='o', label='validation')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
