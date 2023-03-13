from collections.abc import Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
import time


def sigmoid(x,derivative = False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def linear(x,derivative = False):
    if derivative:
        return np.ones_like(x)
    return x




class Layer():
    def __init__(self, input_dim, output_dim, weights=None, bias=None,activation=None):
        # output dim is also the number of neurons in the layer
        self.weights = weights if not weights is None else np.random.uniform(0,1,size= (input_dim, output_dim)) * 0.01
        self.bias =  bias if not bias is None else np.random.uniform(0,1,size=(1, output_dim))
        self.activation = activation if not activation is None else sigmoid
        
    
    def forward(self, inputs):
        self.inputs = inputs
        outputs = np.dot(inputs, self.weights) + self.bias
        self.outputs = self.activation(outputs)
        return self.outputs
    
    
    def backward(self, delta_outputs):
        derivative_activation = self.activation(self.outputs, derivative=True)
        
        delta_inputs = (delta_outputs * derivative_activation) @ self.weights.T
        delta_weights = self.inputs.T @ (delta_outputs * derivative_activation)
        delta_bias = np.sum(delta_outputs * derivative_activation, axis=0, keepdims=True)
        
        return delta_weights, delta_bias, delta_inputs
    
    
    def set_weights(self, weights:np.array):
        self.weights = weights
        
        
    def set_biases(self, biases:np.array):
        self.bias = biases
        
    
    def set_function(self,activation:Callable):
        self.activation = activation

    
    
    def summary(self):
        dict = {
            "weights" : self.weights,
            "biases" : self.bias,
            "function" : self.activation
        }
        return dict
    
        
     
    



class MLP():
    
    def __init__(self, input_dim=None, output_dim=None, hidden_dims=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = []
        if not input_dim is None and not output_dim is None and not hidden_dims is None:
            # create this network automatically if all parameters are given
            self.layers.append(Layer(input_dim, hidden_dims[0], activation=sigmoid))
            for i in range(1, len(hidden_dims)):
                self.layers.append(Layer(hidden_dims[i-1], hidden_dims[i], activation=sigmoid))
            self.layers.append(Layer(hidden_dims[-1], output_dim, activation=linear))
        
    
    def add(self, layer):
        self.layers.append(layer)
        
    
    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs
    
    
    def fit(self, X, y, epochs, batch_size, learning_rate, shuffle=True, loss_stop=0.0001):
        X = np.array(X).reshape(-1, self.input_dim)
        y = np.array(y).reshape(-1, self.output_dim)
        start_time = time.time()
        iter_loss = [mean_squared_error(y, self.predict(X))]
        n_samples = X.shape[0]
        if batch_size == -1 or batch_size > n_samples:
            batch_size = n_samples
            
        n_batches = n_samples // batch_size
        
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            if shuffle:
                permutation = np.random.permutation(n_samples)
                X_ = X[permutation]
                y_ = y[permutation]
            
            
            for batch in range(n_batches):
                batch_start = batch * batch_size
                batch_end = (batch+1) * batch_size
                X_batch = X_[batch_start:batch_end]
                y_batch = y_[batch_start:batch_end]
                
                # forward pass
                outputs = self.predict(X_batch)
                loss = mean_squared_error(y_batch, outputs)
                epoch_loss += loss
                
                # backward pass
                delta_outputs = (outputs - y_batch) / batch_size
                for layer in reversed(self.layers):
                    delta_weights, delta_bias, delta_outputs = layer.backward(delta_outputs)
                    layer.weights -= learning_rate * delta_weights
                    layer.bias -= learning_rate * delta_bias

                
            loss_full = mean_squared_error(y, self.predict(X))
            if loss_full < loss_stop:
                iter_loss.append(loss_full)
                epochs = epoch + 1
                print(f"Loss {loss_full} reached  after {epoch+1} epochs", end="\r")
                return time.time() - start_time, iter_loss, epochs
                
            if (epoch+1)%100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches}",end="\r")
                iter_loss.append(mean_squared_error(y, self.predict(X)))
        return time.time() - start_time, iter_loss, epochs

        
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weights)
        return weights
    

    def get_biases(self):
        biases = []
        for layer in self.layers:
            biases.append(layer.bias)
        return biases


    def summary(self):
        summary_dict = {}
        for i,layer in enumerate(self.layers):
            summary_dict[f"Layer{i}"] = layer.summary()
        return summary_dict
    
    
    def plot_weigths(self):
        weights = self.get_weights()
        biases = self.get_biases()
        

        plt.set_cmap("coolwarm")

        fig, axs = plt.subplots(len(weights), 2, figsize=(16, 8*len(weights)))

        for i, (w, b) in enumerate(zip(weights,biases)):
            # plot weights
            sns.heatmap(w, ax=axs[i][0], square=True, annot=True, fmt=".1f", cbar=False, cmap="bwr", norm=colors.CenteredNorm())
            axs[i][0].set_title(f"Layer {i+1} weights")
            axs[i][0].set_xticklabels(range(1, w.shape[1]+1))
            axs[i][0].set_yticklabels(range(1, w.shape[0]+1))
            
            # plot biases
            sns.heatmap(b, ax=axs[i][1], square=True, annot=True, fmt=".1f", cbar=False, cmap="bwr", norm=colors.CenteredNorm())
            
            axs[i][1].set_title(f"Layer {i+1} biases")
            axs[i][1].set_xticklabels(range(1, b.shape[1]+1))
            axs[i][1].set_yticklabels(range(1, b.shape[0]+1))
        
        
              
#-------------------------Helper functions-------------------------#
        
        
def plot_predictions(mlp:MLP, df_train:pd.DataFrame, df_test:pd.DataFrame):
    #Plots the predictions of the MLP on the training and test data side by side, as well as the MSE of the predictions   
    warnings.filterwarnings('ignore')

    y_train = mlp.predict(df_train[["x"]])
    y_test = mlp.predict(df_test[["x"]])

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))

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
