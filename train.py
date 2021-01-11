import tensorflow as tf
import numpy as np
import keras
from data_generator import load_mnist
from model import full_connect, cnn


def train():
    # set hyperparameters
    batchsize = 64
    epochs = 20
    lr = 0.01
    # get training and testing data
    train_x, train_y = load_mnist('./data', 'train')
    test_x, test_y = load_mnist('./data', 'test')
    # construct model
    model = full_connect()
    model.summary()
    # training
    History = model.fit(train_x, train_y, batch_size=batchsize, epochs=epochs, validation_data=(train_x, train_y),
                        shuffle=True)
    print("History-Train:", History, History.history)
    metrics = model.evaluate(test_x, test_y)
    print("metrics:", metrics)
    # 保存模型
    model.save(save_path)


if __name__ == "__main__":
    train()
