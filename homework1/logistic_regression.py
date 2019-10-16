# -*- coding: utf-8 -*-
# logistic regression on MNIST binary image classification.

# I have completed the major framework, you can fill the space 
# to make it work.
# By the way, achieving the logistic regression with Numpy 
# is a classical coding problem when you confront to 
# look for an intern in some AI labs. 
# So u may treat this as the rehearsal.
# Gooood luck for your first programming assignment. ;)

__author__ = "Zifeng Wang"
__email__  = "wangzf18@mails.tsinghua.edu.cn"
__date__   = "20190920"

import numpy as np
import pdb
import os

# import your own tools
from load_mnist import load_mnist, filter_dataset

np.random.seed(2019)

class logistic_regression:
	"""To develop a classifier, 
	u may build its two basic functions first:
	the train and predict.
	"""
	def __init__(self):
		self.acc_func = lambda x,y: 100 * np.sum(x==y) / x.shape[0]

	def train(self, x_train, y_train):
		"""Receive the input training data, then learn the model.
		Inputs:
		x_train: np.array, shape (num_samples, num_features)
		y_train: np.array, shape (num_samples, )

		Outputs：
		None
		"""
		self.w = np.random.randn(784)
		self.learning_rate = 0.1

		# update the parameters
		for i in range(100):
			pred = self.predict(x_train)
			gradient = np.sum(np.reshape(np.repeat(y_train - pred, x_train.shape[1]), x_train.shape) * x_train, axis=0)
			self.w += self.learning_rate * gradient

	def predict(self, x_test):
		"""Do prediction via the learned model.
		Inputs:
		x_test: np.array, shape (num_samples, num_features)

		Outputs:
		pred: np.array, shape (num_samples, )
		"""
		product = x_test.dot(self.w)
		pred = 1 / (1 + np.exp(-1 * product))
		return pred

if __name__ == '__main__':
	# load data
	x_train, y_train, x_test, y_test = load_mnist()
	x_train, y_train = filter_dataset(x_train, y_train, 1, 7)
	x_test, y_test = filter_dataset(x_test, y_test, 1, 7)
	
	# train ur classifier
	lr = logistic_regression()
	lr.train(x_train, y_train)
	y_test_pred = lr.predict(x_test)

	# evaluate the prediction
	y_test_pred[y_test_pred>0.5] = 1.0
	y_test_pred[y_test_pred<=0.5] = 0.0
	print(np.bincount(y_test_pred.astype(int)))

	test_acc = lr.acc_func(y_test,y_test_pred)
	print("Your model acquires test acc: {:.4f} %".format(test_acc))

	if test_acc >= 95:
		print("Congratulations! Your classifier works!")
	else:
		print("Check your code! Sth went wrong.")
