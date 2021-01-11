import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """
    Load MNIST data from `path`
    """
    assert(kind in ('test', 'train'))
    prefix = 't10k' if kind=='test' else 'train'
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % prefix)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % prefix)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


if __name__ == "__main__":
    x_train, y_train = load_mnist("./data", 'train')
    x_test, y_test = load_mnist('./data', 'test')