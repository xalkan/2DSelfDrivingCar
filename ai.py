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
        samples = zip(*random.sample(self.memory, batch_size))
        # convert samples to torch variables
        # concatenate w.r.t first dimension (states)
        # so that states, action, rewards are aligned
        # Variables contains both tensors and gradients so
        # that weights are updated during gradient descent
        # returns a list of pytorch variables
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

class Dqn():

    # input_size - encoded state as inputs of nn
    # nb_action - possible actions as outputs of nn
    # gamma - discount factor
    # reward_window - mean of the reward over time of last 100 iterations

    # composing our transition events
    # last_state - batch tensor with input_size + 1 dimensions
    # last_action - last action
    # last_reward - last reward
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    # we wont be associating the gradient in this input state with the computation graph
    # of nn module for performance and saving memory
    # Tempareture parameter modulates which action nn decides to play
    # higher - more sure, higher prob of winning q value, lower - less sure
    # or how much exploration vs exploitation ratio we want
    # Here, tempareture is 7
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True)) *100 )
        action = probs.multinomial()    # returns pytorch tensor
        return action.data[0,0]         # returns action

    # TODO get back to this at the end
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # get actions and convert them into vectors from tensors
        # 0 index is state, at index 1 is action
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # next outputs of the next states as used in q learning equation
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # target q values calculated from q learning equation
        target = self.gamma * next_outputs + batch_reward
        # loss function is 
        td_loss = F.smooth_l1_loss(outputs, target)
        # reset the optimizer
        self.optimizer.zero_grad()
        # perform backprop
        td_loss.backward(retain_variables = True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        # new_signal is new state or new observation
        # push new state into memory for experience replay
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]) ))
        # get action based on this new state
        action = self.select_action(new_state)
        # learn
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward  = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        # update action, state and reward
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward

        # reward window size should be fixed
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)      # safety check for dividing by zero

    def save(self):
        # only save nn and optimizer state
        torch.save({'state_dict': self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, 'last_brain.pth')

    def load(self):
        # if last_brain.pth exists load it
        if os.path.isfile('last_brain.pth'):
            print('Loading model...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('no saved model found.')

    
























