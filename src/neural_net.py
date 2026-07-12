import numpy as np
import functions

NEURAL_NET = [
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

def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory


def init_layers(neural_net, seed = 99):
    """
    preconditions:
        input neural network size (no need to input seed - its default)
    postconditions:
        output a neural network with random values for the weights
    """
    np.random.seed(seed) # randomizes seed parameter in function
    # number_of_layers = len(neural_net) # TODO: do we use this?
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
