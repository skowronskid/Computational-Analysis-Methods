from collections.abc import Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, log_loss, f1_score, confusion_matrix
import warnings
import time
from functools import partial



def sigmoid(x,derivative = False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def linear(x,derivative = False):
    if derivative:
        return np.ones_like(x)
    return x


def softmax(x,derivative = False):
    if derivative:
        x = softmax(x)
        return x * (1 - x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def tanh(x, derivative=False):
    if derivative:
        return 1 - np.power(np.tanh(x), 2)
    return np.tanh(x)


def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(x.dtype)
    return np.maximum(x, 0)

def leaky_relu(x, derivative=False, leaky_slope=0.01):
    if derivative:
        return (x > 0).astype(x.dtype) + leaky_slope * (x <= 0).astype(x.dtype)
    return np.maximum(x, leaky_slope * x)
     


class Layer():
    def __init__(self, input_dim, output_dim, weights=None, bias=None,activation=None):
        # output dim is also the number of neurons in the layer
        self.weights = weights if not weights is None else np.random.normal(0, np.sqrt(2/input_dim),size= (input_dim, output_dim)) 
        self.bias =  bias if not bias is None else np.random.normal(0,np.sqrt(2/input_dim),size=(1, output_dim))
        self.activation = activation if not activation is None else sigmoid
        self.momentum_weights = np.zeros_like(self.weights)
        self.momentum_bias = np.zeros_like(self.bias)
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        self.moving_gradient_weights = np.zeros_like(self.weights)
        self.moving_gradient_bias = np.zeros_like(self.bias)        


        
    
    def forward(self, inputs):
        self.inputs = inputs
        outputs = np.dot(inputs, self.weights) + self.bias
        self.outputs = self.activation(outputs)
        return self.outputs
    
    
    def backward(self, delta_outputs,momentum_coef, normalisation_coef):
        derivative_activation = self.activation(self.outputs, derivative=True)
        
        delta_inputs = (delta_outputs * derivative_activation) @ self.weights.T
        delta_weights = self.inputs.T @ (delta_outputs * derivative_activation)
        delta_bias = np.sum(delta_outputs * derivative_activation, axis=0, keepdims=True)

        if normalisation_coef !=0 and momentum_coef !=0:
            # adam optimizer
            epsilon = 1e-8
            self.gradient_weights = delta_weights 
            self.gradient_bias = delta_bias
            
            self.momentum_weights = momentum_coef*self.momentum_weights + (1-momentum_coef)*self.gradient_weights
            self.momentum_bias = momentum_coef*self.momentum_bias + (1-momentum_coef)*self.gradient_bias
            
            self.moving_gradient_weights = normalisation_coef*self.moving_gradient_weights + (1-normalisation_coef)*np.square(self.gradient_weights)
            self.moving_gradient_bias = normalisation_coef*self.moving_gradient_bias + (1-normalisation_coef)*np.square(self.gradient_bias)
            
            delta_weights =  np.divide(self.momentum_weights,np.sqrt(self.moving_gradient_weights) + epsilon)
            delta_bias = np.divide(self.momentum_bias,np.sqrt(self.moving_gradient_bias) + epsilon)
            
            

        elif normalisation_coef !=0:
            epsilon = 1e-8 #prevent division by 0 if is gradient equal to 0

            self.gradient_weights = delta_weights 
            self.gradient_bias = delta_bias
            
            self.moving_gradient_weights = self.moving_gradient_weights*normalisation_coef + (1-normalisation_coef)*np.square(self.gradient_weights)
            self.moving_gradient_bias = self.moving_gradient_bias*normalisation_coef + (1-normalisation_coef)*np.square(self.gradient_bias)

            delta_weights =  np.divide(self.gradient_weights,np.sqrt(self.moving_gradient_weights) + epsilon)
            delta_bias = np.divide(self.gradient_bias,np.sqrt(self.moving_gradient_bias) + epsilon)


        elif momentum_coef != 0:
            self.momentum_weights = delta_weights - momentum_coef*self.momentum_weights
            self.momentum_bias =  delta_bias - momentum_coef*self.momentum_bias
            delta_weights = self.momentum_weights 
            delta_bias = self.momentum_bias

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
    
    def __init__(self, input_dim=None, output_dim=None, hidden_dims=None,loss_function=None, layers_activation = None,classification=False, output_activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = []
        self.classification = classification
        if loss_function is not None:
            self.loss_function =  loss_function
        else:
            self.loss_function = log_loss if classification else mean_squared_error
        if input_dim is not None and  output_dim is not None and  hidden_dims is not None:
            # create this network automatically if all parameters are given
            layers_activation = layers_activation if layers_activation is not None else sigmoid
            self.layers.append(Layer(input_dim, hidden_dims[0], activation=layers_activation))
            for i in range(1, len(hidden_dims)):
                self.layers.append(Layer(hidden_dims[i-1], hidden_dims[i], activation=sigmoid))
            if output_activation is not None:
                output_activation = output_activation
            else:
                output_activation = softmax if classification else linear
            self.layers.append(Layer(hidden_dims[-1], self.output_dim, activation=output_activation))
        
    
    def add(self, layer):
        self.layers.append(layer)
        
    
    def predict(self, inputs, probabilities=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        if not probabilities:
            return np.argmax(outputs, axis=1)
        return outputs
    
    
    def fit(self, X, y, epochs, batch_size, learning_rate, shuffle=True, score_stop=0.0001, momentum_coef=0.9, normalisation_coef=0.999, verbose=True, score_function=None):
        X = np.array(X).reshape(-1, self.input_dim)
        y_true = np.array(y)
        if self.classification:
            y = pd.get_dummies(y)
        y = np.array(y).reshape(-1, self.output_dim)
        start_time = time.perf_counter()
        
        if score_function is None:
            score_function = self.loss_function
        iter_loss = [score_function(y_true, self.predict(X,probabilities=False))]
        times = [0]
        n_samples = X.shape[0]
        if batch_size == -1 or batch_size > n_samples:
            batch_size = n_samples
            
        n_batches = n_samples // batch_size

        if not (momentum_coef >= 0 and momentum_coef < 1):
            raise AttributeError("Momentum decay coefficient should be in [0,1)")
        if not (normalisation_coef >= 0 and normalisation_coef < 1):
            raise AttributeError("Normalisation decay coefficient should be in [0,1)")
        
        
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
                loss = self.loss_function(y_batch, outputs)
                epoch_loss += loss
                
                # backward pass
                delta_outputs = (outputs - y_batch) / batch_size
                for layer in reversed(self.layers):
                    delta_weights, delta_bias, delta_outputs = layer.backward(delta_outputs,momentum_coef=momentum_coef, normalisation_coef=normalisation_coef)                   
                    layer.weights -= learning_rate * delta_weights
                    layer.bias -= learning_rate * delta_bias
            
            probabilities = False if self.classification else True #stupid but that's how I programmed it I guess...
            score_full = score_function(y_true, self.predict(X, probabilities=probabilities))
            if (self.classification and score_full > score_stop) or (not self.classification and score_full < score_stop):
                iter_loss.append(score_full)
                times.append(time.perf_counter() - start_time)
                epochs = epoch + 1
                if verbose:
                    print(f"Score {score_full} reached  after {epoch+1} epochs \n", end="\r")
                return times, iter_loss, epochs
            if (epoch+1)%10 == 0:
                iter_loss.append(score_full)
                times.append( time.perf_counter() - start_time)    
            
            if (epoch+1)%100 == 0 and verbose and self.classification:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches} ,Score: {score_full}",end="\r")
            elif (epoch+1)%100 == 0 and verbose and not self.classification:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches}",end="\r")
                

        return times, iter_loss, epochs
    
        
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
    
    
    def plot_weights(self):
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
    
    
def plot_classification(mlp,df_train,df_test, some_error=False):
    y_pred_test = mlp.predict(df_test[["x","y"]], probabilities=False)
    errors_test = df_test["c"] != y_pred_test

    y_pred = mlp.predict(df_train[["x","y"]],probabilities=False)
    errors = df_train["c"] != y_pred
    warnings.filterwarnings('ignore')

    fig, axes = plt.subplots(2,2,figsize=(10,10))
    if some_error:
        axes[0,0] = sns.scatterplot(x="x", y="y", hue="c", data=df_train, ax=axes[0,0])
        sns.scatterplot(x=df_train["x"][errors], y=df_train["y"][errors], hue=y_pred[errors],style=y_pred[errors], ax=axes[0,0],s=200, markers="x")
        axes[1,0] = sns.scatterplot(x="x", y="y", hue="c", data=df_test, ax=axes[1,0])
        sns.scatterplot(x=df_test["x"][errors_test], y=df_test["y"][errors_test], hue=y_pred_test[errors_test],style=y_pred_test[errors_test], ax=axes[1,0],s=200, markers="x")
        
    else:
        axes[0,0] = sns.scatterplot(x="x", y="y", hue="c", data=df_train, ax=axes[0,0], palette=sns.light_palette("seagreen", as_cmap=True))
        sns.scatterplot(x=df_train["x"][errors], y=df_train["y"][errors], hue=y_pred[errors],style=y_pred[errors], ax=axes[0,0],s=200, markers="x", palette=sns.color_palette("dark:salmon_r", as_cmap=True))
        axes[1,0] = sns.scatterplot(x="x", y="y", hue="c", data=df_test, ax=axes[1,0], palette=sns.light_palette("seagreen", as_cmap=True))
        sns.scatterplot(x=df_test["x"][errors_test], y=df_test["y"][errors_test], hue=y_pred_test[errors_test],style=y_pred_test[errors_test], ax=axes[1,0],s=200, markers="x", palette=sns.color_palette("dark:salmon_r", as_cmap=True))

    axes[0,0].legend()
    axes[0,0].set_title("Training set")
    y_pred = mlp.predict(df_train[["x","y"]],probabilities=False)
    axes[0,1] = sns.heatmap(confusion_matrix(df_train["c"],y_pred), annot=True, fmt="d", cmap="Blues",ax=axes[0,1], cbar=False)
    axes[0,1].set_title(f"F1 score = {f1_score(df_train['c'],y_pred,average='macro'):.3f}")

    axes[1,0].legend()
    axes[1,0].set_title("Test set")
    y_pred = mlp.predict(df_train[["x","y"]],probabilities=False)
    axes[1,1] = sns.heatmap(confusion_matrix(df_test["c"],y_pred_test), annot=True, fmt="d", cmap="Blues",ax=axes[1,1], cbar=False)
    axes[1,1].set_title(f"F1 score = {f1_score(df_test['c'],y_pred_test,average='macro'):.3f}")


    warnings.filterwarnings('default')
    
    
    
def plot_data(title:str, df_train:pd.DataFrame, df_test:pd.DataFrame):
    # Plots the training and test data side by side
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))

    ax[0].scatter(x=df_train["x"], y = df_train['y'])
    ax[0].set_title("Train data")
    ax[1].scatter(x=df_test["x"], y = df_test['y'])
    ax[1].set_title("Test data")

    fig.suptitle(title)
    

def benchmark(x,y,hidden,epochs,range_, learning_rate_base,learning_rate_rmsprop,learning_rate_momentum,learning_rate_adam,momentum_coef,normalisation_coef, adam_coefs, batch_size, loss_stop):
    # this is a benchmark for lab3 regarding different optimizers 
    input = x.shape[1] if len(x.shape) > 1 else 1
    output =  y.shape[1] if len(y.shape) > 1 else 1
    end_time_base = []
    end_time_rmsprop = []
    end_time_momentum = []
    end_time_adam = []
    end_epoch_base = []
    end_epoch_rmsprop = []
    end_epoch_momentum = []
    end_epoch_adam = []
    end_mse_base = []
    end_mse_rmsprop = []
    end_mse_momentum = []
    end_mse_adam = []



    times_base = []
    times_rmsprop = []
    times_momentum = []
    times_adam = []
    mse_base = []
    mse_rmsprop = []
    mse_momentum = []
    mse_adam = []
        
    for i in range(range_):
        print(f"Iteration:{i+1}/{range_}",end="\r") 
        
        #no normalisation or momentum
        mlp1 = MLP(input_dim=input, output_dim=output, hidden_dims=hidden) 
        times1, mse1, epochs1 = mlp1.fit(x, y, epochs=epochs, learning_rate=learning_rate_base,normalisation_coef=0, momentum_coef=0, batch_size=batch_size, loss_stop=loss_stop, verbose=False)
        times_base.append(times1)
        mse_base.append(mse1)
        end_epoch_base.append(epochs1)
        end_time_base.append(times1[-1])
        end_mse_base.append(mse1[-1])
        #rmsprop
        mlp2 = MLP(input_dim=input, output_dim=output, hidden_dims=hidden) 
        times2, mse2, epochs2 = mlp2.fit(x, y, epochs=epochs, learning_rate=learning_rate_rmsprop,normalisation_coef=normalisation_coef, momentum_coef=0, batch_size=batch_size, loss_stop=loss_stop,verbose=False)
        times_rmsprop.append(times2)
        mse_rmsprop.append(mse2)
        end_epoch_rmsprop.append(epochs2)
        end_time_rmsprop.append(times2[-1])
        end_mse_rmsprop.append(mse2[-1])
        #momentum
        mlp3 = MLP(input_dim=input, output_dim=output, hidden_dims=hidden) 
        times3, mse3, epochs3 = mlp3.fit(x, y, epochs=epochs, learning_rate=learning_rate_momentum,normalisation_coef=0, momentum_coef=momentum_coef, batch_size=batch_size, loss_stop=loss_stop,verbose=False)
        times_momentum.append(times3)
        mse_momentum.append(mse3)
        end_epoch_momentum.append(epochs3)
        end_time_momentum.append(times3[-1])
        end_mse_momentum.append(mse3[-1])
        #adam
        mlp4 = MLP(input_dim=input, output_dim=output, hidden_dims=hidden) 
        times4, mse4, epochs4 = mlp4.fit(x, y, epochs=epochs, learning_rate=learning_rate_adam,normalisation_coef=adam_coefs[0], momentum_coef=adam_coefs[1], batch_size=batch_size, loss_stop=loss_stop,verbose=False)
        times_adam.append(times4)
        mse_adam.append(mse4)
        end_epoch_adam.append(epochs4)
        end_time_adam.append(times4[-1])
        end_mse_adam.append(mse4[-1])


    results = pd.DataFrame({"end_time_base":end_time_base,"end_time_rmsprop":end_time_rmsprop,"end_time_momentum":end_time_momentum,"end_time_adam":end_time_adam,
                        "end_epoch_base":end_epoch_base,"end_epoch_rmsprop":end_epoch_rmsprop,"end_epoch_momentum":end_epoch_momentum,"end_epoch_adam":end_epoch_adam,
                        "end_mse_base":end_mse_base,"end_mse_rmsprop":end_mse_rmsprop,"end_mse_momentum":end_mse_momentum,"end_mse_adam":end_mse_adam})
    
    
    # --- Plotting ---
    base_index = results.index[results['end_time_base']==results['end_time_base'].quantile(interpolation='nearest')][0]
    rmsprop_index = results.index[results['end_time_rmsprop']==results['end_time_rmsprop'].quantile(interpolation='nearest')][0]
    momentum_index = results.index[results['end_time_momentum']==results['end_time_momentum'].quantile(interpolation='nearest')][0]
    adam_index = results.index[results['end_time_adam']==results['end_time_adam'].quantile(interpolation='nearest')][0]
    labels = ["base", "rmsprop", "momentum", "adam"]
    base = results.loc[base_index].filter(regex='base').values
    rmsprop = results.loc[rmsprop_index].filter(regex='rmsprop').values
    momentum = results.loc[momentum_index].filter(regex='momentum').values 
    adam = results.loc[adam_index].filter(regex='adam').values

    fig, axes = plt.subplots(1, 2, width_ratios=[1, 3], figsize=(20,8))

    axes[0] = sns.barplot(x=labels,y=[base[0],rmsprop[0],momentum[0],adam[0]], ax=axes[0])
    axes[0].set_title("Median training time" )
    axes[0].set_ylabel("time (s)")



    axes[1] = sns.lineplot(x=times_base[base_index], y = mse_base[base_index], ax=axes[1], label="base")
    axes[1] = sns.lineplot(x=times_rmsprop[rmsprop_index], y = mse_rmsprop[rmsprop_index], ax=axes[1], label="rmsprop")
    axes[1] = sns.lineplot(x=times_momentum[momentum_index], y = mse_momentum[momentum_index], ax=axes[1],label = "momentum")
    axes[1] = sns.lineplot(x=times_adam[adam_index], y = mse_adam[adam_index], ax=axes[1], label="adam")
    axes[1].autoscale(enable=True)
    axes[1].set_yscale("symlog")
    axes[1].set_title("Convergence of the loss function")
    axes[1].set_xlabel("time (s)")

    
    
    
    return results


def coefs_search(X, y,input_dim, output_dim, hidden_dims, layers_activation,classification, epochs, batch_size,score_stop, learning_rates=[0.1,0.01,0.001], momentum_coefs=[0.2,0.5,0.9], normalisation_coefs=[0.9,0.99,0.999], score_function=None, debug=False, allow_warnings=False ):
    warnings.filterwarnings("error") if not allow_warnings else warnings.filterwarnings("ignore")
    for learning_rate in learning_rates:
        for momentum_coef in momentum_coefs:
            for normalisation_coef in normalisation_coefs:
                mlp = MLP(input_dim, output_dim, hidden_dims, layers_activation=layers_activation, classification=classification)
                try:
                    times, loss, epochs =  mlp.fit(X,y,epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, momentum_coef=momentum_coef, normalisation_coef=normalisation_coef, 
                            shuffle=True,score_stop = score_stop, score_function = score_function, verbose=False)
                except Exception as e:
                    if debug:
                        print(e)
                    print(f"Failure for {learning_rate=}, {momentum_coef=}, {normalisation_coef=}")
                else: 
                    print(f"{learning_rate=}, {momentum_coef=}, {normalisation_coef=} |  loss = {loss[-1]} time = {times[-1]}")
                    
    warnings.filterwarnings("default")



def benchmark_architecture(X, y,sample_size,input_dim, output_dim, all_hidden_dims, layers_activation,classification, epochs, batch_size, score_stop, learning_rates, coefs, score_function=None):
    # this is a benchmark for lab5 to compaer the perfomrance of different activation functions and different architectures
    times_list = []
    loss_list = []
    df = pd.DataFrame()
    activation_name = layers_activation.__name__
    for j,hidden_dims, learning_rate, coef in zip(range(len(learning_rates)),all_hidden_dims, learning_rates, coefs):
        end_times_list = []
        end_loss_list = []
        end_epochs_list = []
        for i in range(sample_size):
            print(f"Architecture {j+1}/{len(learning_rates)} - Sample {i+1}/{sample_size}", end="\r")
            mlp = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, layers_activation=layers_activation, classification=classification)
            if score_function is not None:
                times, loss, epochs = mlp.fit(X,y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, momentum_coef=coef[0], normalisation_coef=coef[1], score_stop=score_stop,score_function=score_function, verbose=False)
            else:
                times, loss, epochs = mlp.fit(X,y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, momentum_coef=coef[0], normalisation_coef=coef[1], score_stop=score_stop, verbose=False)
            times_list.append(times)
            loss_list.append(loss)
            end_times_list.append(times[-1])
            end_loss_list.append(loss[-1])
        df[f"end_time_{activation_name}_{len(hidden_dims)}x{hidden_dims[0]}"] = end_times_list
        df[f"end_loss_{activation_name}_{len(hidden_dims)}x{hidden_dims[0]}"] = end_loss_list
    return df, times_list, loss_list

