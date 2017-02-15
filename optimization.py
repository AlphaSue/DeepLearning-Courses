"""All the optimization methods go here.

"""

from __future__ import division, print_function, absolute_import
import random
import numpy as np
import copy

class SGD(object):
    """Mini-batch stochastic gradient descent.

    Attributes:
        learning_rate(float): the learning rate to use.
        batch_size(int): the number of samples in a mini-batch.

    """

    def __init__(self, learning_rate, batch_size):
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size

    def __has_parameters(self, layer):
        return hasattr(layer, "W")

    def compute_gradient(self, x, y, graph, loss):
        """ Compute the gradients of network parameters (weights and biases)
        using backpropagation.

        Args:
            x(np.array): the input to the network.
            y(np.array): the ground truth of the input.
            graph(obj): the network structure.
            loss(obj): the loss function for the network.

        Returns:
            dv_Ws(list): a list of gradients of the weights.
            dv_bs(list): a list of gradients of the biases.

        """

        # TODO: Backpropagation code
        lens=len(graph.config)
        # y_pre = graph.inference(x)
        
        y_pre=x
        for layer in graph[:lens-1]:
                y_pre = layer.forward(y_pre)
                
        dv_Ws=[]
        dv_bs=[]
        
        
        dv_y=loss.backward(y_pre,y)
        
        layers=[layer for layer in graph]
        
        for i in range(lens):
            tempx=copy.copy(x)
            for layer in layers[:lens-i-1]:
                tempx = layer.forward(tempx)
                
            if (lens-i-1)%2 != 0:
                dv_y = layers[lens-i-1].backward(tempx,dv_y)
            else:
                dv_y,dv_W,dv_b = layers[lens-i-1].backward(tempx,dv_y)
                if dv_Ws==[]:
                    dv_Ws=[dv_W]
                else: dv_Ws=[dv_W]+dv_Ws
                if dv_bs==[]:
                    dv_bs=[dv_b]
                else:
                    dv_bs=[dv_b]+dv_bs
                    
        return dv_Ws,dv_bs


    def optimize(self, graph, loss, training_data):
        """ Perform SGD on the network defined by 'graph' using
        'training_data'.

        Args:
            graph(obj): a 'Graph' object that defines the structure of a
                neural network.
            loss(obj): the loss function for the network.
            training_data(list): a list of tuples ``(x, y)`` representing the
                training inputs and the desired outputs.

        """

        # Network parameters
        Ws = [layer.W for layer in graph if self.__has_parameters(layer)]
        bs = [layer.b for layer in graph if self.__has_parameters(layer)]

        # Shuffle the data to make sure samples in each batch are not
        # correlated
        random.shuffle(training_data)
        n = len(training_data)

        batches = [
            training_data[k:k + self.batch_size]
            for k in xrange(0, n, self.batch_size)
        ]

        # TODO: SGD code
        bat=0
        print(len(batches))
        for data in batches:
            bat+=1
            print(bat)
            
            dvW_sum=list(np.zeros(np.array(Ws).shape))
            dvb_sum =list(np.zeros(np.array(bs).shape))
            for data2 in data:
                dv_Ws,dv_bs=self.compute_gradient(list(data2[0]),list(data2[1]),graph,loss)
                dvW_sum=np.add(dvW_sum,dv_Ws)
                dvb_sum=np.add(dvb_sum,dv_bs)
            
            dv_Ws=np.divide(dvW_sum,self.batch_size)
            
            dv_bs=np.divide(dvb_sum,self.batch_size)
            
            # Ws=Ws-np.multiply(self.learning_rate,dv_Ws)
            # bs=bs-np.multiply(self.learning_rate,dv_bs)
            
            layers=[layer for layer in graph]
            
            for i in range(len(layers)):
                if i==0:
                    layers[i].W=layers[i].W-np.multiply(self.learning_rate,dv_Ws[i])
                    layers[i].b=layers[i].b-np.multiply(self.learning_rate,dv_bs[i])
                if i==2:
                    layers[i].W=layers[i].W-np.multiply(self.learning_rate,dv_Ws[1])
                    layers[i].b=layers[i].b-np.multiply(self.learning_rate,dv_bs[1])
            
            
        
