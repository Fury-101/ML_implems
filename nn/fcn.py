# %%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# %%
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Derivative of the sigmoid function.
def gradSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return (abs(x) + x) / 2

import random

class NN():

    def __init__(self, sizes):
        '''
        Initializes the network with an array of sizes representing the neuron counts of each layer.
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Ignores the first input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # List of numpy arrays storing the weights for each connection between adjacent layers.
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, inp):
        '''
        Forward propagation through the network; returns the final prediction for an input.
        '''
        for b, w in zip(self.biases, self.weights):
            inp = sigmoid(np.dot(w, inp) + b)
        return inp

    def train(self, x_train, y_train, epochs, mini_batch_size, lr):
        '''
        Trains the model using stochastic gradient descent (SGD).
        '''
        for j in range(epochs):
            # Randomly shuffle both x_train and y_train in unision
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]
            
            # Divide both x and y into batches for SGD
            x_batches = [
                x_train[k:k+mini_batch_size]
                for k in range(0, len(x_train), mini_batch_size)]
            y_batches = [
                y_train[k:k+mini_batch_size]
                for k in range(0, len(y_train), mini_batch_size)]
            
            for x, y in tqdm(list(zip(x_batches, y_batches))):
                self.handle_batch(x, y, lr)

            print("Epoch {} complete".format(j+1))

    def handle_batch(self, x, y, lr):
        '''
        Handles each batch by computing the gradients for both weights(dCdW) and biases (dCdb) through back propagation & updating them.
        '''
        # Batch gradients for the weights and biases.
        dCdbfull = [np.zeros(b.shape) for b in self.biases]
        dCdWfull = [np.zeros(w.shape) for w in self.weights]
        
        for x_sample, y_sample in zip(x, y):
            dCdb, dCdW = self.backprop(x_sample, y_sample)
            dCdbfull = [nb+dnb for nb, dnb in zip(dCdbfull, dCdb)]
            dCdWfull = [nw+dnw for nw, dnw in zip(dCdWfull, dCdW)]
            
        # Updates the weights and biases with the learning rate and gradient
        self.weights = [w - (lr/len(x_sample))*nw
                        for w, nw in zip(self.weights, dCdWfull)]
        self.biases = [b - (lr/len(x_sample))*nb
                       for b, nb in zip(self.biases, dCdbfull)]

    def backprop(self, x, y):
        ''' 
        Uses back propagation to calculate gradients for weights and biases. Backpropagation explained in further depth below.
        '''
        a = x
        a_history = [x]
        z_history = [] 

        # Forward propagation while recording history
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            z_history.append(z)
            a = sigmoid(z)
            a_history.append(a)
        
        # Following is explained below
        errorL = (a_history[-1] - y) * gradSigmoid(z_history[-1])
        
        dCdW = [np.zeros(w.shape) for w in self.weights]
        dCdb = [np.zeros(b.shape) for b in self.biases]

        dCdW[-1] = np.dot(errorL, a_history[-2].transpose())
        dCdb[-1] = errorL


        for l in range(self.num_layers-1, 1, -1):
            z = z_history[l-2]
            sp = gradSigmoid(z)
            errorL = np.dot(self.weights[l-1].transpose(), errorL) * sp
            dCdW[-l] = np.dot(errorL, a_history[l-2].transpose())
            dCdb[-l] = errorL
        return (dCdb, dCdW)

    def evaluate(self, test_data):
        '''
        Returns the fraction of correct predictions of the model on the provided test data.
        '''
        test_results = [(np.argmax(self.forward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) / len(test_results)


# %% [markdown]
# ### Backpropagation internals
# Our cost function is $C = \frac{1}{2} \sum \left( y_i - a^{(L)}_i \right)^2$ (where $a^{(L)}_i$ denotes the activation of the neuron $i$ of layer $L$).
# 
# Let $e^{(l)}_i = \frac{\partial C}{\partial z_j^{(l)}}$ be the error at layer $l$, neuron $i$. Let $L$ be the last layer. The vector $$e^{(L)} = \frac{\partial C}{\partial a^{(L)}} \frac{\partial a^{(L)}}{\partial z^{(L)}}$$ by chain rule (applied for each element).
# $$ e^{(L)} = \frac{\partial C}{\partial a^{(L)}} \frac{\partial \sigma (z^{(L)})}{\partial z^{(L)}} $$
# $$ e^{(L)} = (a^{(L)} - y) \sigma'(z^{(L)}) $$
# The rest of the algorithm just propagates the error back through the network. Notably, the transpose of the weights is used instead of the actual weights.

# %% [markdown]
# 


