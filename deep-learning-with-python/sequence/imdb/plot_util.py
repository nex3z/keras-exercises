import matplotlib.pylab as plt
from keras.callbacks import History


def plot_history(history: History, metrics=('acc',)) -> None:
    history_dict = history.history
    for metric in metrics:
        train_metric = history_dict[metric]
        val_metric = history_dict['val_{}'.format(metric)]
        epochs = range(1, len(train_metric) + 1)

        plt.figure()
        plt.plot(epochs, train_metric, label='Training {}'.format(metric))
        plt.plot(epochs, val_metric, label='Validation {}'.format(metric))
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.show()
