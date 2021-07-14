# train.py

import dvclive
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils


class MetricsCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            dvclive.log(metric, value)
        dvclive.next_step()


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    classes = 10
    y_train = np_utils.to_categorical(y_train, classes)
    y_test = np_utils.to_categorical(y_test, classes)
    return (x_train, y_train), (x_test, y_test)


def get_model():
    model = Sequential()

    model.add(Dense(512, input_dim=784))
    model.add(Activation("relu"))
    model.add(Dense(10, input_dim=512))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="sgd"
    )
    return model


(x_train, y_train), (x_test, y_test) = load_data()
model = get_model()

# dvclive.init("training_metrics")  # Implicit with DVC
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=128,
    epochs=3,
    callbacks=[MetricsCallback()],
)
