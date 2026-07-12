import numpy as np
import functions

NN_CONFIG = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    """
    A_prev is the input signal (probably)
    """
    Z_curr = np.dot(W_curr, A_prev) + b_curr # batch apply multiplications dont matter

    if activation == "relu":
        activation_func = functions.relu
    elif activation == "sigmoid":
        activation_func = functions.sigmoid
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr # f(x), x of the activation function

def full_forward_propagation(X, params_values, nn_config):
    memory = {} # dictionary
    A_curr = X # input value (initial f(x) output)

    for idx, layer in enumerate(nn_config):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)] # fetch layer weights from nn
        b_curr = params_values["b" + str(layer_idx)] # fetch layer biases from nn
        # grabs the input and output of the current function based on previous node and weights
        # activation(x), x
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        memory["A" + str(idx)] = A_prev # matches the keys from nn, updates the value
        memory["Z" + str(layer_idx)] = Z_curr # same here
    return A_curr, memory # returns the output and the a(x) and x to store in cache for backprop

def get_cost_value(Y_hat, Y):
    """
    precondtions:
        input your model predicted value and actual value
    postconditions:
        average loss
    """
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = round(Y_hat) # converts decimal from 0.0 -> 1.0 to 0 or 1
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1] # amount of samples (amount of columns it has)

    if activation == "relu":
        backward_activation_func = functions.relu_backward
    elif activation == "sigmoid":
        backward_activation_func = functions.sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m # change in cost with respect to W
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m # change in cost with respect to b
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_config):
    grads_values = {}
    # m = Y.shape[1] # unused
    Y = Y.reshape(Y_hat.shape) # safety??

    # change in loss with difference of predicted and actual
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_config))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev # change in loss with difference (the punishment)

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr # all mentions of dN were referencing cost function
        grads_values["db" + str(layer_idx_curr)] = db_curr
    return grads_values

def init_layers(nn_config, seed = 99):
    """
    preconditions:
        input neural network size (no need to input seed - its default)
    postconditions:
        output a neural network with random values for the weights
    """
    np.random.seed(seed) # randomizes seed parameter in function
    # number_of_layers = len(neural_net) # TODO: do we use this?
    params_values = {} # no values yet

    for idx, layer in enumerate(nn_config): # for each layer in the network
        layer_idx = idx + 1 # so its not 0 indexed?
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1 # creates random weights
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1 # creates some random biases

    return params_values
