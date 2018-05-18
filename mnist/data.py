import keras
from keras.datasets import mnist

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CLASSES = 10


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return pre_process(x_train, y_train), pre_process(x_test, y_test)
    # x_train = x_train.reshape(x_train.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    # x_test = x_test.reshape(x_test.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255
    # y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    # y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    # return (x_train, y_train), (x_test, y_test)


def pre_process(x, y):
    x = x.reshape(x.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    x = x.astype('float32') / 255
    y = keras.utils.to_categorical(y, NUM_CLASSES)
    return x, y
