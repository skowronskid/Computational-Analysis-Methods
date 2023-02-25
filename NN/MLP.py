from collections.abc import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings


def sigmoid(x,derivative = False):
    if derivative:
        sigm = sigmoid(x,False)
        return sigm * (1 - sigm)
    return 1 / (1 + np.exp(-x))


def linear(x,derivative = False):
    if derivative:
        return np.ones(x.shape)
    return x


class Layer():      
    """_summary_
    A layer class for a multi-layer perceptron
    
    _parameters_
    nr_neurons: int
        The number of neurons in the layer. 
    weights: np.array
        The weights of the layer. The shape of the array should be (nr_neurons, nr_inputs). Each row represents the weights of one neuron.
    biases: np.array
        The biases of the layer. The shape of the array should be (nr_neurons, 1). Each row represents the bias of one neuron.
    func: Callable
        The activation function of the layer. The function should take a numpy array as input and return a numpy array as output.
    
    _methods_
    forward(X)
        Calculates the output of the layer for the given input X and returns it. 
    summary()
        Returns a dictionary containing the weights, biases and function of the layer. 
    set_weights(weights)
        Sets the weights of the layer.
    set_biases(biases)
        Sets the biases of the layer
    set_function(func)
        Sets the function of the layer
    set_nr_neurons(nr_neurons)
        Sets the number of neurons of the layer
    """
    
    
    def __init__(self, nr_neurons, weights:np.array=None, biases:np.array=None, func:Callable=None):
        if not (weights is None and biases is None) and nr_neurons!=len(weights)!=len(biases):
            raise Warning("Incorrect number neurons or shapes of either weights or biases")
              
        
        self.nr_neurons = nr_neurons
        self.biases = biases
        self.weights = weights
        self.function = func
        
        # for backpropagation
        self.z_value = None
        self.delta = None
        
        
    def forward(self, X:np.array):
        if any(el is None for el in [self.weights, self.biases, self.function]):
            raise AttributeError("Something in this layer is missing")
        if len(X.shape) == 1:
            X.reshape(1, X.shape[0])
        
        self.z_value = X @ self.weights.T + self.biases
        return self.function(self.z_value)
    

    def backward(self, X, y):
        # X is activation of previous aka it's forward output
        # y is the true val? at least for the last layer
        if self.function == linear:
            self.delta = (self.function(self.z_value) - y)
        else:
            self.delta = (self.function(self.z_value, derivative=True)) @ (self.function(self.z_value) - y)
        self.delta_W =  X.T @ self.delta 
        self.delta_b = np.sum(self.delta, axis=0, keepdims=True)
         ## TODO....    
    

    
    
    def summary(self):
        dict = {
            "weights" : self.weights,
            "biases" : self.biases,
            "function" : self.function
        }
        return dict
    
    
    def set_weights(self, weights:np.array):
        if weights.shape[0] != self.nr_neurons:
            raise Warning("The number of neurons on this layer seems to be different to the number of weights given. Maybe use reshape to state the shape explicitly")
        self.weights = weights
        
        
    def set_biases(self, biases:np.array):
        # if len(biases) != self.nr_neurons:
            # raise Warning("The number of neurons on this layer seems to be different to the number of biases given.")
        self.biases = biases
        
    
    def set_function(self,func:Callable):
        self.function = func
        
    
    def set_nr_neurons(self,nr_neurons:int):
        self.nr_neurons = nr_neurons
        
     
    



class MLP():           
    """_summary_
    A multi-layer perceptron class
    
    _parameters_
    layers: dict
        A dictionary containing the layers of the MLP. The keys are the names of the layers and the values are the layers themselves.
    weights: list
        A list containing the weights of the MLP. The weights are stored in the same order as the layers in the layers dictionary.
    biases: list
        A list containing the biases of the MLP. The biases are stored in the same order as the layers in the layers dictionary.
    
    _methods_
    add(layer:Layer, name:str=None)
        Adds a layer to the MLP. The name of the layer is optional. If no name is given, the name of the layer will be 'Layer' + the number of layers already in the MLP.
    predict(inputs)
        Predicts the output for the given input.
    summary()
        Returns a dictionary containing the weights, biases and functions of all layers in the MLP.
    """
    
    
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
    
    
    def backpropagation(self, y_true):
        layers = list(self.layers.values())
        for i in range(len(layers)-1, -1, -1):
            layers[i].backward(y_true)
        ## TODO....    
        


    def summary(self):
        summary_dict = {}
        for name,layer in zip(self.layers.keys(), self.layers.values()):
            summary_dict[name] = layer.summary()
        return summary_dict
    

    def pop(self):
        if len(self.layers) == 0:
            raise Warning("There are no layers in this MLP")
        self.layers.popitem()
        self.weights.pop()
        self.biases.pop()
        
        
              
#-------------------------Helper functions-------------------------#
        
        
def plot_predictions(mlp:MLP, df_train:pd.DataFrame, df_test:pd.DataFrame):
    #Plots the predictions of the MLP on the training and test data side by side, as well as the MSE of the predictions   
    warnings.filterwarnings('ignore')

    y_train = mlp.predict(df_train[["x"]])
    y_test = mlp.predict(df_test[["x"]])

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))

    ax[0].set_title("Training data (MSE: " + str(mean_squared_error(df_train[["y"]], y_train)) + ")",fontsize=10)
    ax[0].scatter(df_train["x"],df_train["y"], color="blue")
    ax[0].scatter(df_train['x'], y_train, color='red')

    ax[1].set_title("Test data (MSE: " + str(mean_squared_error(df_test[["y"]], y_test))+ ")",fontsize=10)
    ax[1].scatter(df_test["x"],df_test["y"], color="blue")
    ax[1].scatter(df_test['x'], y_test, color='red')

    fig.suptitle("blue: true values, red: predicted values", fontsize=12)
    plt.subplots_adjust(top=0.87)
    warnings.filterwarnings('default')
    
    
def plot_data(title:str, df_train:pd.DataFrame, df_test:pd.DataFrame):
    # Plots the training and test data side by side
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))

    ax[0].scatter(x=df_train["x"], y = df_train['y'])
    ax[0].set_title("Train data")
    ax[1].scatter(x=df_test["x"], y = df_test['y'])
    ax[1].set_title("Test data")

    fig.suptitle(title)
