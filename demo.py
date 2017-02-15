"""This is a simple example for training a neural network. This example
demonstrates some usage of the modules. You may use this code to test your
implementation.

"""

import mnist_loader
import numpy as np
from graph import Graph
from loss import Euclidean
from network import Network
from optimization import SGD

# Load the MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# The network definition of a neural network
graph_config = [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]

graph = Graph(graph_config)
loss = Euclidean()
optimizer = SGD(3.0, 10)  # learning rate 3.0, batch size 10
network = Network(graph)

# Train the network for 1 epoch
global epochs
epochs=1
network.train(training_data, 1, loss, optimizer, test_data)

x, y = test_data[0]
y_pred = np.argmax(network.inference(x))
# Test a handwritten digit image
# sum=0
# for i in np.arange(np.array(test_data).shape[0]):
    # x, y = test_data[i]
    # y_pred = np.argmax(network.inference(x))
    # print(y,y_pred)
    # if y_pred==y:
        # sum=sum+1
# print("the accuracy is : %d" % (sum/np.array(test_data).shape[0]))
print("The image is {}. Your prediction is {}.".format(y, y_pred))
