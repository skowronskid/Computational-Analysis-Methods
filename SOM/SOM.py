
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



# neighbourhood functions
def gaussian(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2))


def gaussian_second_derivative(x, sigma):
    return -(-1 / sigma**2) * (x / sigma)**2 * np.exp(-(x**2) / (2 * sigma**2))


# learning rate decay function
def learning_decay(learning_rate, epoch, decay_coef):
    return learning_rate * np.exp(-epoch / decay_coef)


class SOM:
    def __init__(self, input_dim, map_size, topology='rectangular', X=None):
        self.input_dim = input_dim
        self.map_size = map_size
        self.topology = topology
        if topology == 'rectangular':
            self.weights = np.random.randn(*map_size, input_dim)
        elif topology == 'hexagonal':
            if X is None:
                raise ValueError('X must be provided for hexagonal topology')
            self.weights = np.random.uniform(np.min(X), np.max(X), size=(map_size[0]*map_size[1], input_dim))
        else:
            raise ValueError('Topology must be either "rectangular" or "hexagonal"')
        
        
        
        

    def train(self, X, epochs, neighborhood_function = gaussian, learning_rate=0.1, sigma=None,learning_rate_decay=None, verbose=False):
        if sigma is None:
            sigma = np.max(self.map_size)
        if learning_rate_decay is None:
            learning_rate_decay = epochs
        
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}",end="\r")
            current_learning_rate = learning_decay(learning_rate, epoch, learning_rate_decay)
            permutation = np.random.permutation(X.shape[0])
            X_ = X[permutation]
            
            
            for x in X_:
                               
                
                
                # Compute the distances between the input vector and all neurons
                distances = np.linalg.norm(self.weights - x, axis=-1)
                
                # Find the index of the winning neuron
                winner = np.unravel_index(np.argmin(distances), distances.shape)
                
                # Compute the neighborhood function centered around the winner
                neighbor_indices = np.indices(self.map_size)
                neighbor_distances = np.linalg.norm(neighbor_indices - np.array(winner)[:, np.newaxis, np.newaxis], axis=0)
                neighborhood = neighborhood_function(neighbor_distances, sigma)
                
                # Update the weights of all neurons based on the neighborhood function and learning rate
                self.weights += current_learning_rate * neighborhood[..., np.newaxis] * (x - self.weights)
                
    
    
    def find_clusters(self, X):
        # Compute distances between input vectors and all neurons
        winners = []
        for x in X:
                # Compute the distances between the input vector and all neurons
            distances = np.linalg.norm(self.weights - x, axis=-1)
            
            # Find the index of the winning neuron
            winner = np.argmin(distances)
            winners.append(winner)

        return winners
