from collections.abc import Callable
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear(x):
    return x


class Layer():      
    
    
    def __init__(self, nr_neurons, weights:np.array=None, biases:np.array=None, func:Callable=None):
        if not (weights is None and biases is None) and nr_neurons!=len(weights)!=len(biases):
            raise Warning("Incorrect number neurons or shapes of either weights or biases")
              
        
        self.nr_neurons = nr_neurons
        self.biases = biases
        self.weights = weights
        self.function = func
        
        
    def forward(self, X):
        if any(el is None for el in [self.weights, self.biases, self.function]):
            raise AttributeError("Something in this layer is missing")
        if len(X.shape) == 1:
            X.reshape(1, X.shape[0])
        
        Z = X @ self.weights.T + self.biases
        return self.function(Z)
    
    
    def summary(self):
        dict = {
            "weights" : self.weights,
            "biases" : self.biases,
            "function" : self.function
        }
        return dict
    
    
    def set_weights(self, weights):
        if weights.shape[0] != self.nr_neurons:
            raise Warning("The number of neurons on this layer seems to be different to the number of weights given. Maybe use reshape to state the shape explicitly")
        self.weights = weights
        
        
    def set_biases(self, biases):
        if len(biases) != self.nr_neurons:
            raise Warning("The number of neurons on this layer seems to be different to the number of biases given.")
        self.biases = biases
        
    
    def set_function(self,func):
        self.function = func
        
    
    def set_nr_neurons(self,nr_neurons):
        self.nr_neurons = nr_neurons
        
     
    



class MLP():           
    
    
    def __init__(self):
        self.layers = {}
        self.weights = []
        self.biases = []

    
    def add(self, layer:Layer, name:str = None) -> None:
        if name is None:
            name = f'Layer{len(self.layers)}'
        if name in self.layers.keys():
            raise KeyError(f"A layer with name {name} is already in this MLP")
        
        self.layers[name] = layer
        self.weights.append(layer.weights)
        self.biases.append(layer.biases)
        
        
    def predict(self,inputs):
        for layer in self.layers.values():
            inputs = layer.forward(inputs)
        return inputs      
    
    
    def summary(self):
        summary_dict = {}
        for name,layer in zip(self.layers.keys(), self.layers.values()):
            summary_dict[name] = layer.summary()
        return summary_dict
              
        