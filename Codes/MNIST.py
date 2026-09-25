import numpy as np
import pandas as pd

#LOAD THE DATA
print("Loading training data")
data = pd.read_csv('mnist_train.csv')
data = np.array(data)

# m = images, n = 785 (1 label + 784 pixels)
m, n = data.shape 

# Shuffle the data
np.random.shuffle(data)

Y = data[:, 0]
X = data[:, 1:n].T 
X = X / 255.0

#Neural Network Parameters
input_size = 784
hidden_size = 128
output_size = 10


def init_para():
    W1 = np.random.rand(hidden_size, input_size) - 0.5
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.rand(output_size, hidden_size) - 0.5
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # Subtracting max(Z) to prevent errors with large exponentials
    exp_Z = np.exp(Z - np.max(Z, axis=0)) 
    return exp_Z / np.sum(exp_Z, axis=0)

def one_hot(Y):
    # Creates the perfect answer key (0s and 1s)
    one_hot_Y = np.zeros((output_size, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1 
    return one_hot_Y

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    
    dZ2 = A2 - one_hot_Y
    dW2 = np.dot(dZ2, A1.T) / m  
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dZ1_raw = np.dot(W2.T, dZ2)
    dZ1 = dZ1_raw * (Z1 > 0) # ReLU derivative 
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

#Train the model
def gradient_descent(X, Y, alpha, epochs):
    W1, b1, W2, b2 = init_para()
    
    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Print accuracy every 50 loops
        if i % 50 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Epoch {i} | Accuracy: {accuracy * 100:.2f}%")
            
    return W1, b1, W2, b2

print("Starting training")
W1, b1, W2, b2 = gradient_descent(X, Y, alpha=1.0, epochs=500)
# print("w1 =", W1)
# print("b1 =", b1)
# print("w2 =", W2)
# print("b2 =", b2)
print("Training complete!")

#------------------------------------------------------------------------------------------

#Testing the model
print("Loading test data...")
test_data = pd.read_csv('mnist_test.csv').values

# Split labels and pixels, then normalize
Y_test = test_data[:, 0]
X_test = test_data[:, 1:].T / 255.0

# Run forward propagation trained weights
_, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)

# Get predictions and calculate final accuracy
test_predictions = get_predictions(A2_test)
test_accuracy = get_accuracy(test_predictions, Y_test)

print(f" Test Accuracy: {test_accuracy * 100:.2f}%")
