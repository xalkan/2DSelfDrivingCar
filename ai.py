# Deep Q Network for 2D Self Driving Car

# Library imports
import torch                        # Primary
import torch.nn as nn               # Neural Network
import torch.nn.functional as F     # For Loss function (ReLU)
import torch.optim as optim         # Gradient Descent optimizer
import torch.autograd as autograd   # Convergence from tensors variable
from torch.autograd import Variable # to gradient variable
import numpy as numpy               # Secondary
import random                       # Native
import os

# NN Architecture
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        # number of neurons in the input and output layers
        self.input_size = input_size
        self.hidden_layer_neurons = 30
        self.nb_action = nb_action
        # full connection between the input layer and hidden layer with 30 neurons
        self.full_connection_one = nn.Linear(self.input_size, self.hidden_layer_neurons)
        # full connection between the hidden layer and output layer
        self.full_connection_two = nn.Linear(self.hidden_layer_neurons, self.nb_action)

    # states in, q values out
    def forward(self, state):
        # x = hidden neurons
        x = F.relu(self.full_connection_one(state))
        q_values = self.full_connection_two(x)
        return q_values

# Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
         # maximum numbers of memories/past experiences
         self.capacity = capacity
         # a list of last events
         self.memory = []

    # append new event in memory
    # make sure number of events in memory = capacity
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # returns uniform distribution of random samples where size = batch_size
    def sample(self, batch_size):
        # zip is like reshape function because we need to transform
        # [ [state1, action1, reward1], [state2, reward2, action2] ] to
        # [ [state1, state2], [action1, action2], [reward1, reward2] ]
        samples = zip(*random.sample(self.memory, batch_size)
        # convert samples to torch variables
        # concatenate w.r.t first dimension (states)
        # so that states, action, rewards are aligned
        # Variables contains both tensors and gradients so
        # that weights are updated during gradient descent
        # returns a list of pytorch variables
        return map(lambda x: Variable(torch.cat(x, 0)), samples)





