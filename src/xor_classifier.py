import numpy as np

epochs: int = 60000

input_layer_size, hidden_layer_size, output_layer_size = 2, 3, 1
learning_rate = 0.1

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

w_hidden = np.random.uniform(size=(input_layer_size, hidden_layer_size))
w_output = np.random.uniform(size=(hidden_layer_size, output_layer_size))

def sigmoid(x) -> float: return 1 / ( 1 + np.exp(-x) )
def d_sigmoid(x) -> float: return x * ( 1 - x )

for epoch in range(epochs):
    # forward
    activation_hidden = sigmoid(np.dot(X, w_hidden))
    output = np.dot(activation_hidden, w_output)

    # logging
    error = y - output

    if epoch % 5000 == 0:
        print(f"current errors: {sum(error)}")

    # backwards
    dZ = error * learning_rate
    w_output += activation_hidden.T.dot(dZ) # .T is the transpose of a matrix ( flips on side )
    dH = dZ.dot(w_output.T) * d_sigmoid(activation_hidden)
    w_hidden += X.T.dot(dH)
