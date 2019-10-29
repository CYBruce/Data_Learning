# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:14:36 2018 [V1]
Edited on 21/10/9102 to reformulate the code in Pytorch style. [V2]

@author1: Tianwen Xie
@author2: Zifeng Wang
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') 
import pdb
import datetime


"[ATTENTION] There are total 5 spaces for you to fill."


# ----------
# Forget it, it is an empty class.
# ----------
class Module:
    def __init__(self):
        pass
    def forward(self):
        pass

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

# ----------
# Build the main NN structure.
# ----------
class SimpleNN(Module):
    """Build a simple Neural network with one hidden layer.
    This is generally a Pytorch-style of building a NN, 
    that is, you can only define the forward function for the NN in this class. 
    But here without Pytorch, we have to achieve the 
    backward() function for it on our own ;)
    """
    def __init__(self, learning_rate):
        super(SimpleNN, self).__init__()
        #  initilize the components needed for the NN.
        self.linear = Linear(1,80)
        self.activation = ReLU()
        self.output = Linear(80,1)
        # initilize the learning rate
        self.lr = learning_rate

    def forward(self, inputs, is_training=False):
        """Do forward inference.
        Args:
            inputs: a numpy.array, shape [# of samples, 1]
            is_training: a bool.
        Outputs:
            out: a numpy.array, shape [# of samples, 1]
        """
        x1 = self.linear(inputs)
        x2 = self.activation(x1)
        out = self.output(x2)

        if not is_training:
            return out
        else:
            # if is_training==True, return the hidden outputs for backward.
            return x1, x2, out

    def backward(self, x_train, y_train):
        """Do forward inference, and update the parameters with BP.
        Args:
            x_train: a numpy.array, shape [# of samples, 1]
            y_train: a numpy.array, shape [# of samples, 1]
        Outputs:
            loss: computed loss.
        """
        x1, x2, out = self.forward(x_train, is_training=True)
        loss , loss_grad = self._mean_square_loss(out, y_train)

        # Now start to use your backward function to update the parameters.
        # Dont forget transfer the self.lr to the backward function.
        # * = self.output.backward(*)
        # * = self.activation.backward(*)
        # self.linear.backward(*)

        """
        Please Fill Your Code Here.
        """
        grad_x2 = self.output.backward(x2, loss_grad, self.lr)
        grad_x1 = self.activation.backward(x1, grad_x2)
        self.linear.backward(x_train, grad_x1, self.lr)

        return loss

    def _mean_square_loss(self, y_pred, y_true):
        """Compute the loss function.
        """
        loss = np.power(y_pred - y_true, 2).mean()*0.5
        loss_grad = (y_pred - y_true)/y_pred.shape[0]
        return loss , loss_grad

# ----------
# Build the components required by the NN.
# ----------
class Linear(Module):
    def __init__(self,input_dim,output_dim):
        super(Linear, self).__init__()
        # initilize weights
        self.W = np.random.randn(input_dim,output_dim) * 1e-2
        self.b = np.zeros((1,output_dim))
                       
    def forward(self,inputs):
        """
        Args:            
            inputs: a numpy.array, shape [# of samples, 1]

        Outputs:
            outputs: a numpy.array, shape [# of samples, 1]
        """
        """
                Please Fill Your Code Here.
        """
        outputs = np.dot(inputs, self.W) + self.b

        return outputs
    
    def backward(self, inputs, grad_out, lr):
        """Do backpropagation , update weights in this step.
        """
        
        # self.W -= lr * delt_W
        # self.b -= lr * delt_b
        # and return something for other layers.

        """
        Please Fill Your Code Here.
        """
        outputs = np.dot(grad_out,self.W.T)
        print(type(inputs))
        self.W -= lr * np.dot(inputs.T, grad_out)
        self.b -= lr * np.sum(grad_out,axis=0)
        return outputs

class ReLU(Module):
    # ReLu layer
    def __init__(self):
        super(ReLU, self).__init__()
        pass

    def forward(self,inputs):
        """
        Args:            
            inputs: a numpy.array, shape [# of samples, 1]

        Outputs:
            outputs: a numpy.array, shape [# of samples, 1]
        """

        """
        Please Fill Your Code Here.
        """
        outputs = np.maximum(inputs, 0)
        # return  *

        return outputs

    def backward(self,inputs, grad_output, lr=None):
        """No parameter required to be updated for ReLU,
        but it needs to transfer the gradients to other layers.
        """

        """
        Please Fill Your Code Here.
        """
        # return *
        outputs = np.where(inputs>0, 1, 0) * grad_output
        
        return outputs



if __name__ == '__main__':
    # generate the data
    x = np.linspace(-np.pi,np.pi,140).reshape(140,-1)
    y = np.sin(x)

    #set learning rate
    lr = 0.02

    # build the model
    model = SimpleNN(lr)

    # count steps and save loss history
    loss = 1
    step = 0
    l= []

    # training
    while loss >= 1e-4 and step < 15000:
        loss = model.backward(x, y)
        l.append(loss)
        print("{}/15000  loss: {:.8f}".format(step+1,loss))
        step += 1
        
    # after training , plot the results
    y_pre = model(x)
    plt.plot(x,y,c='r',label='true_value')
    plt.plot(x,y_pre,c='b',label='predict_value')
    plt.legend()
    plt.title(datetime.datetime.now().ctime())
    plt.savefig("1.png")
    plt.figure()
    plt.plot(np.arange(0,len(l)), l )
    plt.title('loss history')
    plt.show()
