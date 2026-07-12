import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    # A = f(x) where x is Z
    # dA is the derivative of the previous node without parameters applied
    # Z is the output of the previous node after parameters applied
    # TODO: verify this after we use sigmoid_backward
    """
    backwards for backwards propagation
    """
    sig = sigmoid(Z)
    d_sig = sig * (1 - sig)
    return dA * d_sig # change in loss with change in Z (pre-activation value (x))

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ
