"""
This file implements a constructive cascade network for the imagemanipulation-eyegaze data set. It is designed
to solve the classification problem about whether the image is manipulated based on the eye gaze on it.
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
input_data = data.drop(['image', 'image manipulated', 'vote'], axis=1)
target_data = data['image manipulated']

# normalise input data using z-score
for column in range(input_data.shape[1]):
    temp = input_data.iloc[:, column]
    input_data.iloc[:, column].apply(lambda x: (x - temp.mean()) / temp.std())

"""
Step 2: Define a constructive cascade network
There are 5 input features, 
a number of cascades with 3 hidden units,
and 2 outputs classes (manipulated/not manipulated)
"""

# maximum number of cascade layers
maximum_cascade_number = 5

input_neuron = input_data.shape[1]
cascade_neuron = 3
output_neuron = 2
learning_rate = 0.01
num_epochs = 800


class ConstructiveCascadeNetwork(torch.nn.Module):
    def __init__(self, in_neuron, cascade_no, out_neuron, hidden_neuron):
        super(ConstructiveCascadeNetwork, self).__init__()
        print("cascades", "+" * 20, cascade_no)
        self.out = nn.Linear(in_neuron + cascade_no * hidden_neuron, out_neuron)
        # store the cascades in a modulelist
        self.cascades = nn.ModuleList()
        for i in range(cascade_number):
            layer = torch.nn.Linear(input_neuron + i * hidden_neuron, hidden_neuron)
            # load previous weight into the corresponding cascade layers
            if parameters and i < cascade_number - 1:
                layer.weight = nn.Parameter(parameters['cascades.' + str(i) + '.weight'])
                layer.bias = nn.Parameter(parameters['cascades.' + str(i) + '.bias'])
            self.cascades.append(layer)

    def forward(self, x):
        # create a copy as we will modify the input later, for debugging purpose
        self.inputs = x.clone()
        for layer_number in range(len(self.cascades)):
            # Add the newly generated output to the input of the next cascade layer
            c_input = self.cascades[layer_number](self.inputs)
            c_output = torch.sigmoid(c_input)
            self.inputs = torch.cat((self.inputs, c_output), dim=1)
        y_pred = self.out(self.inputs)
        return y_pred


# define loss function
loss_func = nn.CrossEntropyLoss()

# no. of runs of the network
no_of_runs = 20

# record the average test accuracy over runs
average_test_accuracy = 0

# record the structure of the result by recording the no. of cascades
structure_record = {0:0,1:0,2:0,3:0,4:0,5:0}

for i in range(no_of_runs):

    # Halt criterion: A global variable for termination of training, terminates
    # if the model fails to decrease validation error after addition of 2 cascades
    terminate = False

    # A global variable for recording the weights and bias
    parameters = None

    cascade_number = 1

    # randomly split data into training set (60%), validation set (20%) and testing set (20%)
    msk = np.random.rand(len(data)) < 0.6

    # split training data into input and target
    # all but the second last column are inputs, the second last one is target
    train_input = input_data[msk]
    train_target = target_data[msk]

    # All the data left
    other_input = input_data[~msk]
    other_target = target_data[~msk]

    msk2 = np.random.rand(len(other_input)) < 0.5
    # split validation data into input and target
    # all but the second last column are inputs, the second last one is target
    validation_input = other_input[msk2]
    validation_target = other_target[msk2]

    # split test data into input and target
    # all but the second last column are inputs, the second last one is target
    test_input = other_input[~msk2]
    test_target = other_target[~msk2]

    # create Tensors to hold training inputs and outputs
    X = torch.Tensor(train_input.values.astype(float))
    Y = torch.Tensor(train_target.values.astype(float)).long()

    # create Tensors to hold validation inputs and outputs
    X_validate = torch.Tensor(validation_input.values.astype(float))
    Y_validate = torch.Tensor(validation_target.values.astype(float)).long()

    previous_two_validation_errors = [float("inf"), float("inf")] # Record validation loss of two previous networks

    while not terminate and cascade_number <= maximum_cascade_number:

        # if it is true, add a cascade layer
        increase_cascade = False

        net = ConstructiveCascadeNetwork(input_neuron, cascade_number, output_neuron, cascade_neuron)

        optimiser = torch.optim.Rprop(net.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):

            Y_pred = net(X)

            loss = loss_func(Y_pred, Y)

            # # print progress
            # if epoch % 200 == 0:
            #     # convert two-column predicted Y values to one column for comparison
            #     _, predicted = torch.max(Y_pred, 1)
            #
            #     # calculate and print accuracy
            #     total = predicted.size(0)
            #     correct = predicted.data.numpy() == Y.data.numpy()
            #
            #     print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
            #           % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))

            net.zero_grad()

            loss.backward()

            optimiser.step()

            if loss < 0.2 or epoch == num_epochs - 1:

                # the condition for adding cascade is met
                cascade_number += 1
                # copy the weights and bias for loading in the next iteration of network construction
                parameters = net.state_dict()

                Y_pred_validate = net(X_validate)
                # Compute validation loss
                loss = loss_func(Y_pred_validate, Y_validate)

                _, predicted = torch.max(Y_pred_validate, 1)

                # calculate and print accuracy
                total = predicted.size(0)
                correct = predicted.data.numpy() == Y_validate.data.numpy()

                print('validation Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
                      % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))

                # Test if the validation loss fails to decrease with the addition of 2 cascades. If so, terminate
                if loss < max(previous_two_validation_errors):
                    # if the termination condition is not met, update the validation errors stored
                    previous_two_validation_errors.pop(0)
                    previous_two_validation_errors.append(loss)
                else:
                    terminate = True
                break

    # record the final structure of the network(in terms of no. of cascades)
    structure_record[cascade_number - 1] += 1


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
    _, predicted_test = torch.max(Y_pred_test, 1)

    # calculate accuracy
    total_test = predicted_test.size(0)
    correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

    average_test_accuracy += (correct_test/total_test)

    print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

# calculate average testing accuracy over 20 runs
average_test_accuracy /= no_of_runs
print("average_test_accuracy: %.2f %%" % (average_test_accuracy * 100))

# print the structures of the final network in the 20 runs
print(structure_record)
