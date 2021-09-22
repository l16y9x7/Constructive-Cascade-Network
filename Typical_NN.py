"""
This file is for comparison with the constructive cascade network only.
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

"""
Step 1: load data
"""
# load all data
data = pd.read_excel('Caldwell_ImageManipulation-EyeGaze_DataSetCombined.xlsx',
                     sheet_name='data')

# data.drop(data.columns[0], axis=1, inplace=True)

input_data = data.drop(['image','image manipulated', 'vote'], axis=1)
target_data = data['image manipulated']

# normalise input data
for column in range(input_data.shape[1]):
    temp = input_data.iloc[:, column]
    input_data.iloc[:, column].apply(lambda x: (x - temp.mean()) / temp.std())

# randomly split data into training set (80%), testing set (20%)
msk = np.random.rand(len(data)) < 0.8

# split training data into input and target
# all but the second last column are inputs, the second last one is target
train_input = input_data[msk]
train_target = target_data[msk]

# All the data left
test_input = input_data[~msk]
test_target = target_data[~msk]

# create Tensors to hold training inputs and outputs
X = torch.Tensor(train_input.values.astype(float))
Y = torch.Tensor(train_target.values.astype(float)).long()

X_test = torch.Tensor(test_input.values.astype(float))
Y_test = torch.Tensor(test_target.values.astype(float)).long()

input_neuron = input_data.shape[1]
output_neuron = 3
learning_rate = 0.01
num_epochs = 800


class LogisticRegression(torch.nn.Module):

    def __init__(self, in_neuron,out_neuron):
        super(LogisticRegression, self).__init__()
        self.hidden1 = nn.Linear(in_neuron, 5)
        self.hidden2 = nn.Linear(5, 5)
        self.out = nn.Linear(5, out_neuron)

    def forward(self, x):
        hidden1 = torch.sigmoid(self.hidden1(x))
        hidden2 = torch.sigmoid(self.hidden2(hidden1))
        output = self.out(hidden2)
        return output


# define loss function
loss_func = nn.CrossEntropyLoss()

# define a neural network using the customised structure
net = LogisticRegression(input_neuron, output_neuron)

# define optimiser
optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

for epoch in range(num_epochs):

    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X)
    # Compute loss
    loss = loss_func(Y_pred, Y)
    all_losses.append(loss.item())

    # print progress
    if epoch % 50 == 0:
        # convert two-column predicted Y values to one column for comparison
        _, predicted = torch.max(Y_pred, 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

plt.figure()
plt.plot(all_losses)
plt.show()

"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""

# create Tensors to hold inputs and outputs
X_test = torch.Tensor(test_input.values.astype(float))
Y_test = torch.Tensor(test_target.values.astype(float)).long()

# test the neural network using testing data
Y_pred_test = net(X_test)

# get prediction
# convert predicted Y values to one column for comparison
_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))