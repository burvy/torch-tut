import numpy as np
import src.functions as fns

NEURAL_NET = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation is "relu":
        activation_func = fns.relu
    elif activation is "sigmoid":
        activation_func = fns.sigmoid
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr


def init_layers(neural_net, seed = 99):
    np.random.seed(seed) # randomizes seed parameter in function
    number_of_layers = len(neural_net) # length of neural net
    params_values = {} # no values yet

    for idx, layer in enumerate(neural_net): # for each layer in the network
        layer_idx = idx + 1 # so its not 0 indexed?
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1 # creates random weights
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1 # creates some random biases

    return params_values

print(init_layers(NEURAL_NET))
