from os import name
from keras.layers.core import Activation
from keras.models import Sequential
from keras import optimizers
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Dense, Flatten, Dropout, Lambda, Reshape, GlobalAveragePooling2D


def full_connect(lr=0.01):
    model = Sequential()
    model.add(Dense(50, input_shape=(784,), activation='relu', name='layer1'))
    model.add(Dense(50, activation='relu', name='layer2'))
    model.add(Dense(10, activation='softmax', name='layer3'))
    model.summary()
    rmsprop = optimizers.RMSprop(lr)
    sgd = optimizers.sgd(lr=lr, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model
    

def full_connect_5():
    model = Sequential()
    model.add(Dense(400, input_shape=(784,), activation='relu', name='layer1'))
    model.add(Dense(400, activation='relu', name='layer2'))
    model.add(Dense(400, activation='relu', name='layer3'))
    model.add(Dense(10, activation='softmax', name='layer5'))
    model.summary()
    rmsprop = optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
    return model


def full_connect_8():
    model = Sequential()
    model.add(Dense(300, input_shape=(784,), activation='relu', name='layer1'))
    model.add(Dense(300, activation='relu', name='layer2'))
    model.add(Dense(300, activation='relu', name='layer3'))
    model.add(Dense(300, activation='relu', name='layer4'))
    model.add(Dense(300, activation='relu', name='layer5'))
    model.add(Dense(300, activation='relu', name='layer6'))
    model.add(Dense(10, activation='softmax', name='layer7'))
    model.summary()
    rmsprop = optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
    return model


def cnn(lr=0.01):
    model = Sequential()
    model.add(Reshape((28,28,1), input_shape=(784,)))
    model.add(Conv2D(16, (3, 3), activation="relu", padding="valid", name="block1_conv1"))
    model.add(Conv2D(16, (3, 3), activation="relu", padding="valid", name="block1_conv2"))
    model.add(MaxPooling2D((2,2), name='block1_mp'))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="valid", name="block2_conv1"))
    model.add(MaxPooling2D((2,2), name='block2_mp'))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="valid", name="block3_conv1"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="valid", name="block3_conv2"))
    model.add(Flatten())
    model.add(Dense(20, activation='relu', name='fc1'))
    model.add(Dense(10, activation='softmax', name='fc2'))
    model.summary()
    sgd = optimizers.sgd(lr=lr, momentum=0.9, nesterov=True)
    rmsprop = optimizers.RMSprop(lr)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
    return model


def cnn_gap(lr=0.01):
    model = Sequential()
    model.add(Reshape((28,28,1), input_shape=(784,)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="valid", name="block1_conv1"))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="valid", name="block1_conv2"))
    model.add(MaxPooling2D((2,2), name='block1_mp'))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="valid", name="block2_conv1"))
    model.add(Conv2D(10, (3, 3), activation="relu", padding="valid", name="block2_conv2"))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    model.summary()
    sgd = optimizers.sgd(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


if __name__ == "__main__":
    # fc_net = full_connect()
    # fc_net5 = full_connect_5()
    # fc_net8 = full_connect_8()
    # cnn1 = cnn()
    cnn2 = cnn_gap()