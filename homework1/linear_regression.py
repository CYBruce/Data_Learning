# -*- coding: utf-8 -*-
# linear regression.

# I have completed the major framework, you can fill the space 
# to make it work.

__author__ = "Zifeng Wang"
__email__  = "wangzf18@mails.tsinghua.edu.cn"
__date__   = "20190920"

import numpy as np
import pdb
import os

np.random.seed(2019)

class linear_regression:

	def __init__(self):
		self.mse_func = lambda x,y: 1/x.shape[0] * np.sum((x-y)**2)
		pass

	def train(self,x_train, y_train):
		"""Receive the input training data, then learn the model.
		Inputs:
		x_train: np.array, shape (num_samples, num_features)
		y_train: np.array, shape (num_samples, )

		Outputs：
		None
		"""

		self.learning_rate = 1e-6 # 0.0001, I changed the learning rate
		self.w = np.random.rand(10)
		iteration = 500 # 10000, I changed # of iterations

		for i in range(iteration):
			pred = self.predict(x_train)
			error = pred - y_train
			error_brod = np.reshape(np.repeat(error, x_train.shape[1]), x_train.shape)
			element_wise_product = error_brod * x_train
			gradient = np.sum(element_wise_product, axis=0)
			# gradient = np.sum(np.reshape(np.repeat(pred - y_train, x_train.shape[1]), x_train.shape) * x_train, axis=0)

			self.w = self.w - self.learning_rate * gradient
			if i % 100 == 0:
				print("Iteration {}/{}, MSE loss {:.4f}".format(i+1, iteration, self.mse_func(pred, y_train)))

		return

	def predict(self, x_test):
		"""Do prediction via the learned model.
		Inputs:
		x_test: np.array, shape (num_samples, num_features)

		Outputs:
		pred: np.array, shape (num_samples, )
		"""
		
		pred = x_test.dot(self.w)

		return pred



if __name__ == '__main__':
	
	# "load" your data
	real_weights = np.random.rand(10)
	x_train = np.random.randn(10000,10) * 10
	y_train = x_train.dot(real_weights) + np.random.randn(10000)

	# train and test your model
	linear_regressor = linear_regression()
	linear_regressor.train(x_train, y_train)

	weights_error = np.sum((real_weights - linear_regressor.w)**2)

	if weights_error < 1e-4:
		print("Congratulations! Your linear regressor works!")
	else:
		print("Check your code! Sth went wrong.")
