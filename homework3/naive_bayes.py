# -*- coding: utf-8 -*-

__author__ = "Zifeng Wang"
__email__  = "wangzf18@mails.tsinghua.edu.cn"
__date__   = "20191005"

import numpy as np
import pdb
from sklearn.naive_bayes import BernoulliNB

np.random.seed(2019)

class BernoulliNaiveBayes:
    def __init__(self):
        return

    def train(self, x_train, y_train):
        """Learn the model.
        Inputs:
            x_train: np.array, shape (num_samples, num_features)
            y_train: np.array, shape (num_samples, )

        Outputs：
            None
        """

        """
        Please Fill Your Code Here.
        """

        # learn the p(y=1) and p(y=0)
        num_y1 = sum(y_train)
        num_y0 = len(y_train) - num_y1
        self.p_y1 = num_y1 / len(y_train)
        self.p_y0 = 1 - self.p_y1

        # learn the p(x|y=0), p(x|y=1), they should be a vector with shape (# of feature, ).
        # Laplace smoothing
        y1_set = x_train[np.where(y_train == 1)]
        y0_set = x_train[np.where(y_train == 0)]
        self.p_x_y0 = (np.sum(y0_set, axis=0)+1)/(num_y0+2)
        self.p_x_y1 = (np.sum(y1_set, axis=0)+1)/(num_y1+2)


    def predict(self, x_test):
        """Do prediction via the learnt model.
        Inputs:
            x_test: np.array, shape (num_samples, num_features)

        Outputs:
            pred: np.array, shape (num_samples, )
        """
        x_test = x_test.astype(int)
        p0 = np.sum(np.log(self.p_x_y0 * x_test + (1 - self.p_x_y0) * (1 - x_test)), axis=1) + np.log(self.p_y0)
        p1 = np.sum(np.log(self.p_x_y1 * x_test + (1 - self.p_x_y1) * (1 - x_test)), axis=1) + np.log(self.p_y1)
        pred = np.where((p1 - p0) > 0, 1, 0)

        # do inference to obtain p(y=0|x) and p(y=1|x),
        # then compare them to get the prediction.

        return pred

def load_data(data_path="a1a.txt"):
    labels = []
    x = None
    with open(data_path, "r") as f:
        for i,line in enumerate(f.readlines()):
            if i % 200 == 0:
                print("Processing line No.{}.".format(i))
            data_list = line.split()
            label = (int(data_list[0]) + 1) / 2
            feature_idx = [int(l.split(":")[0]) for l in data_list[1:]]
            labels.append(label)
            features = np.zeros(123)
            features[feature_idx] = 1.0

            if x is None:
                x = features
            else:
                x = np.c_[x,features]

    x = x.T
    labels = np.array(labels).astype(int)
    all_idx = np.arange(x.shape[0])
    np.random.shuffle(all_idx)
    x_train = x[all_idx[:1000]]
    y_train = labels[all_idx[:1000]]
    x_test  = x[all_idx[1000:]]
    y_test  = labels[all_idx[1000:]]

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    print("# of Training data:",x_train.shape[0])
    print("# of Test data:",x_test.shape[0])

    acc_func = lambda x,y: 100 * (x == y).sum() / y.shape[0]

    # clf = BernoulliNB()
    # clf.fit(x_train, y_train)
    clf = BernoulliNaiveBayes()

    clf.train(x_train, y_train)
    pred = clf.predict(x_test)
    # pred = clf.predict(x_test)

    test_acc = acc_func(pred, y_test)
    print("You model acquires Test Acc:{:.2f} %".format(test_acc))

    if test_acc > 75:
        print("Congratulations! Your Naive Bayes classifier WORKS!")
    else:
        print("Check your code! Sth went wrong.")
