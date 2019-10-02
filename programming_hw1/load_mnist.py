# -*- coding: utf-8 -*-
# This is a tool for loading compressed MNIST dataset.
# You can easily transfer it to your own project to play on MNIST dataset. ;)

__author__ = "Zifeng Wang"
__email__  = "wangzf18@mails.tsinghua.edu.cn"

import numpy as np
import pdb
import os

# load mnist dataset
def load_mnist(validation_size = 5000):
    import gzip
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder(">")
        return np.frombuffer(bytestream.read(4),dtype=dt)[0]

    def extract_images(f):
        print("Extracting",f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf,dtype=np.uint8)
            data = data.reshape(num_images,rows,cols,1)
            return data
    
    def extract_labels(f):
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    data_dir = "./data"
    TRAIN_IMAGES = os.path.join(data_dir,'train-images-idx3-ubyte.gz')
    with open(TRAIN_IMAGES,"rb") as f:
        train_images = extract_images(f)

    TRAIN_LABELS =  os.path.join(data_dir,'train-labels-idx1-ubyte.gz')
    with open(TRAIN_LABELS,"rb") as f:
        train_labels = extract_labels(f)

    TEST_IMAGES =  os.path.join(data_dir,'t10k-images-idx3-ubyte.gz')
    with open(TEST_IMAGES,"rb") as f:
        test_images = extract_images(f)

    TEST_LABELS =  os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz')
    with open(TEST_LABELS,"rb") as f:
        test_labels = extract_labels(f)

    # split train and val
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    # preprocessing
    train_images = train_images.astype(np.float32) / 255
    test_images  = test_images.astype(np.float32) / 255
    
    # reshape for logistic regression
    train_images = np.reshape(train_images, [train_images.shape[0], -1])
    test_images = np.reshape(test_images, [test_images.shape[0], -1])
    return train_images,train_labels,test_images,test_labels

def filter_dataset(X, Y, pos_class, neg_class, mode=None):
    """
    Filters out elements of X and Y that aren't one of pos_class or neg_class
    then transforms labels of Y so that +1 = pos_class, -1 = neg_class.
    """
    assert(X.shape[0] == Y.shape[0])
    assert(len(Y.shape) == 1)

    Y = Y.astype(int)
    
    pos_idx = Y == pos_class
    neg_idx = Y == neg_class        
    Y[pos_idx] = 1
    Y[neg_idx] = -1
    idx_to_keep = pos_idx | neg_idx
    X = X[idx_to_keep, ...]
    Y = Y[idx_to_keep]
    if Y.min() == -1 and mode != "svm":
        Y = (Y + 1) / 2
        Y.astype(int)
    return (X, Y)