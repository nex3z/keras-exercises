import matplotlib.pylab as plt
from IPython.display import SVG, display
from keras.callbacks import History
from keras.models import Model
from keras.utils.vis_utils import model_to_dot


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


def show_model(model: Model, show_shapes=True) -> None:
    display(SVG(model_to_dot(model, show_shapes=show_shapes).create(prog='dot', format='svg')))
