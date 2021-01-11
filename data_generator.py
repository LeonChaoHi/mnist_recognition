import os
import struct
import numpy as np
from keras.utils import np_utils

def load_mnist(path, kind='train'):
    """
    Load MNIST data from `path`
    """
    assert(kind in ('test', 'train'))
    prefix = 't10k' if kind=='test' else 'train'
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % prefix)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % prefix)
    with open(labels_path, 'rb') as label:
        magic, n = struct.unpack('>II', label.read(8))
        data_y = np.fromfile(label, dtype=np.uint8)

    with open(images_path, 'rb') as img:
        magic, num, rows, cols = struct.unpack('>IIII', img.read(16))
        data_x = np.fromfile(img, dtype=np.uint8).reshape(-1, 784)

    data_x = data_x.astype('float32') / 255
    data_y = np_utils.to_categorical(data_y, 10)
    
    return data_x, data_y


if __name__ == "__main__":
    x_train, y_train = load_mnist("./data", 'train')
    x_test, y_test = load_mnist('./data', 'test')