import tensorflow as tf
import numpy as np
import keras
from data_generator import load_mnist
from model import full_connect, cnn, cnn_gap


def train():
    # set hyperparameters
    batchsize = 200
    epochs = 10
    lr = 0.1
    # get training and testing data
    train_x, train_y = load_mnist('./data', 'train')
    test_x, test_y = load_mnist('./data', 'test')
    # construct model
    model = cnn_gap(lr)
    # training
    History = model.fit(train_x, train_y, batch_size=batchsize, epochs=epochs, validation_data=(test_x, test_y),
                        shuffle=True)
    print("History-Train:", History, History.history)
    # metrics = model.evaluate(test_x, test_y)
    # print("metrics:", metrics)
    accuracy = eval_accuracy(model.predict(test_x), test_y)
    print("accuracy:", accuracy)
    # 保存模型
    save_path = './models/model1.h5'
    model.save(save_path)
    

def eval_accuracy(pred_y, ground_truth_y):
    pred_y = pred_y.argmax(axis=1).astype('int16')
    ground_truth_y = ground_truth_y.argmax(axis=1).astype('int16')
    accuracy = np.mean(np.where(pred_y==ground_truth_y, np.ones_like(pred_y), np.zeros_like(pred_y)))
    return accuracy


if __name__ == "__main__":
    train()
