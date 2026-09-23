import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# ============================================================
# 1. LOAD MNIST DATA
# ============================================================

print("Loading training data...")

train_data = pd.read_csv("mnist_train.csv").values
test_data = pd.read_csv("mnist_test.csv").values

# Shuffle training data
np.random.seed(42)
np.random.shuffle(train_data)

# Training data
Y_train = train_data[:, 0].astype(int)
X_train = train_data[:, 1:].T / 255.0

# Test data
Y_test = test_data[:, 0].astype(int)
X_test = test_data[:, 1:].T / 255.0


print("Training data shape:", X_train.shape)
print("Training labels shape:", Y_train.shape)

print("Test data shape:", X_test.shape)
print("Test labels shape:", Y_test.shape)


# ============================================================
# 2. NEURAL NETWORK PARAMETERS
# ============================================================

INPUT_SIZE = 784
OUTPUT_SIZE = 10


# ============================================================
# 3. INITIALIZE PARAMETERS
# ============================================================

def init_para(hidden_size):

    # Same random starting point for fair comparison
    np.random.seed(42)

    # Input -> hidden layer
    W1 = np.random.randn(
        hidden_size,
        INPUT_SIZE
    ) * np.sqrt(2 / INPUT_SIZE)

    b1 = np.zeros(
        (hidden_size, 1)
    )

    # Hidden -> output layer
    W2 = np.random.randn(
        OUTPUT_SIZE,
        hidden_size
    ) * np.sqrt(2 / hidden_size)

    b2 = np.zeros(
        (OUTPUT_SIZE, 1)
    )

    return W1, b1, W2, b2


# ============================================================
# 4. RELU
# ============================================================

def ReLU(Z):

    return np.maximum(0, Z)


# ============================================================
# 5. SOFTMAX
# ============================================================

def softmax(Z):

    # Subtract maximum to prevent overflow
    exp_Z = np.exp(
        Z - np.max(
            Z,
            axis=0,
            keepdims=True
        )
    )

    return exp_Z / np.sum(
        exp_Z,
        axis=0,
        keepdims=True
    )


# ============================================================
# 6. ONE-HOT ENCODING
# ============================================================

def one_hot(Y):

    one_hot_Y = np.zeros(
        (OUTPUT_SIZE, Y.size)
    )

    one_hot_Y[
        Y,
        np.arange(Y.size)
    ] = 1

    return one_hot_Y


# ============================================================
# 7. GET PREDICTIONS
# ============================================================

def get_predictions(A2):

    return np.argmax(
        A2,
        axis=0
    )


# ============================================================
# 8. GET ACCURACY
# ============================================================

def get_accuracy(predictions, Y):

    return np.mean(
        predictions == Y
    )


# ============================================================
# 9. CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(A2, Y):

    m = Y.size

    correct_probabilities = A2[
        Y,
        np.arange(m)
    ]

    loss = -np.mean(
        np.log(
            correct_probabilities + 1e-12
        )
    )

    return loss


# ============================================================
# 10. FORWARD PROPAGATION
# ============================================================

def forward_prop(
    W1,
    b1,
    W2,
    b2,
    X
):

    # Hidden layer
    Z1 = np.dot(W1, X) + b1

    A1 = ReLU(Z1)

    # Output layer
    Z2 = np.dot(W2, A1) + b2

    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


# ============================================================
# 11. BACKPROPAGATION
# ============================================================

def backward_prop(
    Z1,
    A1,
    A2,
    W2,
    X,
    Y
):

    # Number of images
    m = X.shape[1]

    one_hot_Y = one_hot(Y)

    # -------------------------
    # Output layer
    # -------------------------

    dZ2 = A2 - one_hot_Y

    dW2 = np.dot(
        dZ2,
        A1.T
    ) / m

    db2 = np.sum(
        dZ2,
        axis=1,
        keepdims=True
    ) / m


    # -------------------------
    # Hidden layer
    # -------------------------

    dZ1_raw = np.dot(
        W2.T,
        dZ2
    )

    # ReLU derivative
    dZ1 = dZ1_raw * (Z1 > 0)

    dW1 = np.dot(
        dZ1,
        X.T
    ) / m

    db1 = np.sum(
        dZ1,
        axis=1,
        keepdims=True
    ) / m

    return dW1, db1, dW2, db2


# ============================================================
# 12. UPDATE PARAMETERS
# ============================================================

def update_params(
    W1,
    b1,
    W2,
    b2,
    dW1,
    db1,
    dW2,
    db2,
    alpha
):

    W1 = W1 - alpha * dW1

    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2

    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


# ============================================================
# 13. TRAIN MODEL
# ============================================================

def gradient_descent(
    X,
    Y,
    alpha,
    epochs,
    hidden_size,
    show_progress=False
):

    W1, b1, W2, b2 = init_para(
        hidden_size
    )

    loss_history = []
    accuracy_history = []

    start_time = time.time()


    for epoch in range(epochs):

        # -----------------------
        # Forward propagation
        # -----------------------

        Z1, A1, Z2, A2 = forward_prop(
            W1,
            b1,
            W2,
            b2,
            X
        )


        # -----------------------
        # Calculate loss
        # -----------------------

        loss = cross_entropy_loss(
            A2,
            Y
        )


        # -----------------------
        # Calculate accuracy
        # -----------------------

        predictions = get_predictions(
            A2
        )

        accuracy = get_accuracy(
            predictions,
            Y
        )


        loss_history.append(loss)

        accuracy_history.append(
            accuracy
        )


        # -----------------------
        # Backpropagation
        # -----------------------

        dW1, db1, dW2, db2 = backward_prop(
            Z1,
            A1,
            A2,
            W2,
            X,
            Y
        )


        # -----------------------
        # Update weights
        # -----------------------

        W1, b1, W2, b2 = update_params(
            W1,
            b1,
            W2,
            b2,
            dW1,
            db1,
            dW2,
            db2,
            alpha
        )


        if show_progress:

            if epoch % 10 == 0:

                print(
                    f"Epoch {epoch:4d} | "
                    f"Loss = {loss:.4f} | "
                    f"Accuracy = {accuracy * 100:.2f}%"
                )


    training_time = (
        time.time() - start_time
    )


    # Calculate final training values

    _, _, _, A2_train = forward_prop(
        W1,
        b1,
        W2,
        b2,
        X
    )

    final_train_predictions = get_predictions(
        A2_train
    )

    final_train_accuracy = get_accuracy(
        final_train_predictions,
        Y
    )

    final_train_loss = cross_entropy_loss(
        A2_train,
        Y
    )


    return (
        W1,
        b1,
        W2,
        b2,
        final_train_accuracy,
        final_train_loss,
        training_time,
        loss_history,
        accuracy_history
    )


# ============================================================
# 14. TEST MODEL
# ============================================================

def test_model(
    W1,
    b1,
    W2,
    b2
):

    _, _, _, A2_test = forward_prop(
        W1,
        b1,
        W2,
        b2,
        X_test
    )

    predictions = get_predictions(
        A2_test
    )

    accuracy = get_accuracy(
        predictions,
        Y_test
    )

    loss = cross_entropy_loss(
        A2_test,
        Y_test
    )

    return accuracy, loss


# ============================================================
# ============================================================
#
# EXPERIMENT 1
# LEARNING RATE VS ACCURACY
#
# ============================================================
# ============================================================

print("\n")
print("=" * 60)
print("EXPERIMENT 1: LEARNING RATE")
print("=" * 60)


learning_rates = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.5
]


alpha_train_accuracy = []
alpha_test_accuracy = []

alpha_train_loss = []
alpha_test_loss = []

alpha_training_time = []


for alpha in learning_rates:

    print(
        f"\nTesting learning rate = {alpha}"
    )

    (
        W1,
        b1,
        W2,
        b2,
        train_accuracy,
        train_loss,
        training_time,
        losses,
        accuracies
    ) = gradient_descent(

        X_train,
        Y_train,

        alpha=alpha,

        # Fixed
        epochs=50,

        # Fixed
        hidden_size=128
    )


    test_accuracy, test_loss = test_model(
        W1,
        b1,
        W2,
        b2
    )


    alpha_train_accuracy.append(
        train_accuracy * 100
    )

    alpha_test_accuracy.append(
        test_accuracy * 100
    )


    alpha_train_loss.append(
        train_loss
    )

    alpha_test_loss.append(
        test_loss
    )


    alpha_training_time.append(
        training_time
    )


    print(
        f"Training Accuracy = "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Test Accuracy = "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Loss = "
        f"{test_loss:.4f}"
    )


# ============================================================
# GRAPH 1
# LEARNING RATE VS ACCURACY
# ============================================================

plt.figure()

plt.plot(
    learning_rates,
    alpha_train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    learning_rates,
    alpha_test_accuracy,
    marker="o",
    label="Test Accuracy"
)

plt.xlabel(
    "Learning Rate (Alpha)"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Learning Rate vs Accuracy"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# GRAPH 2
# LEARNING RATE VS LOSS
# ============================================================

plt.figure()

plt.plot(
    learning_rates,
    alpha_test_loss,
    marker="o"
)

plt.xlabel(
    "Learning Rate (Alpha)"
)

plt.ylabel(
    "Test Loss"
)

plt.title(
    "Learning Rate vs Test Loss"
)

plt.grid()

plt.show()



# ============================================================
# ============================================================
#
# EXPERIMENT 2
# EPOCHS VS ACCURACY
#
# ============================================================
# ============================================================

print("\n")
print("=" * 60)
print("EXPERIMENT 2: NUMBER OF EPOCHS")
print("=" * 60)


epoch_values = [
    10,
    25,
    50,
    100,
    200
]


epoch_train_accuracy = []
epoch_test_accuracy = []

epoch_train_loss = []
epoch_test_loss = []

epoch_training_time = []


for epochs in epoch_values:

    print(
        f"\nTesting epochs = {epochs}"
    )


    (
        W1,
        b1,
        W2,
        b2,
        train_accuracy,
        train_loss,
        training_time,
        losses,
        accuracies
    ) = gradient_descent(

        X_train,
        Y_train,

        # Fixed
        alpha=0.1,

        epochs=epochs,

        # Fixed
        hidden_size=128
    )


    test_accuracy, test_loss = test_model(
        W1,
        b1,
        W2,
        b2
    )


    epoch_train_accuracy.append(
        train_accuracy * 100
    )

    epoch_test_accuracy.append(
        test_accuracy * 100
    )


    epoch_train_loss.append(
        train_loss
    )

    epoch_test_loss.append(
        test_loss
    )


    epoch_training_time.append(
        training_time
    )


    print(
        f"Training Accuracy = "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Test Accuracy = "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Loss = "
        f"{test_loss:.4f}"
    )


# ============================================================
# GRAPH 3
# EPOCHS VS ACCURACY
# ============================================================

plt.figure()

plt.plot(
    epoch_values,
    epoch_train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epoch_values,
    epoch_test_accuracy,
    marker="o",
    label="Test Accuracy"
)

plt.xlabel(
    "Number of Epochs"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Epochs vs Accuracy"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# GRAPH 4
# EPOCHS VS LOSS
# ============================================================

plt.figure()

plt.plot(
    epoch_values,
    epoch_test_loss,
    marker="o"
)

plt.xlabel(
    "Number of Epochs"
)

plt.ylabel(
    "Test Loss"
)

plt.title(
    "Epochs vs Test Loss"
)

plt.grid()

plt.show()



# ============================================================
# ============================================================
#
# EXPERIMENT 3
# HIDDEN NEURONS VS ACCURACY
#
# ============================================================
# ============================================================

print("\n")
print("=" * 60)
print("EXPERIMENT 3: HIDDEN LAYER NEURONS")
print("=" * 60)


hidden_neuron_values = [
    16,
    32,
    64,
    128
]


hidden_train_accuracy = []
hidden_test_accuracy = []

hidden_train_loss = []
hidden_test_loss = []

hidden_training_time = []


for hidden_size in hidden_neuron_values:

    print(
        f"\nTesting hidden neurons = "
        f"{hidden_size}"
    )


    (
        W1,
        b1,
        W2,
        b2,
        train_accuracy,
        train_loss,
        training_time,
        losses,
        accuracies
    ) = gradient_descent(

        X_train,
        Y_train,

        # Fixed
        alpha=0.1,

        # Fixed
        epochs=50,

        hidden_size=hidden_size
    )


    test_accuracy, test_loss = test_model(
        W1,
        b1,
        W2,
        b2
    )


    hidden_train_accuracy.append(
        train_accuracy * 100
    )

    hidden_test_accuracy.append(
        test_accuracy * 100
    )


    hidden_train_loss.append(
        train_loss
    )

    hidden_test_loss.append(
        test_loss
    )


    hidden_training_time.append(
        training_time
    )


    print(
        f"Training Accuracy = "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Test Accuracy = "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Training Time = "
        f"{training_time:.2f} seconds"
    )


# ============================================================
# GRAPH 5
# HIDDEN NEURONS VS ACCURACY
# ============================================================

plt.figure()

plt.plot(
    hidden_neuron_values,
    hidden_train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    hidden_neuron_values,
    hidden_test_accuracy,
    marker="o",
    label="Test Accuracy"
)

plt.xlabel(
    "Number of Hidden Neurons"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Hidden Neurons vs Accuracy"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# GRAPH 6
# HIDDEN NEURONS VS TRAINING TIME
# ============================================================

plt.figure()

plt.plot(
    hidden_neuron_values,
    hidden_training_time,
    marker="o"
)

plt.xlabel(
    "Number of Hidden Neurons"
)

plt.ylabel(
    "Training Time (seconds)"
)

plt.title(
    "Hidden Neurons vs Training Time"
)

plt.grid()

plt.show()



# ============================================================
# 15. PRINT RESULT TABLES
# ============================================================

print("\n")
print("=" * 60)
print("LEARNING RATE RESULTS")
print("=" * 60)


learning_rate_results = pd.DataFrame({

    "Learning Rate":
        learning_rates,

    "Train Accuracy (%)":
        alpha_train_accuracy,

    "Test Accuracy (%)":
        alpha_test_accuracy,

    "Test Loss":
        alpha_test_loss,

    "Training Time (s)":
        alpha_training_time
})


print(
    learning_rate_results.to_string(
        index=False
    )
)



print("\n")
print("=" * 60)
print("EPOCH RESULTS")
print("=" * 60)


epoch_results = pd.DataFrame({

    "Epochs":
        epoch_values,

    "Train Accuracy (%)":
        epoch_train_accuracy,

    "Test Accuracy (%)":
        epoch_test_accuracy,

    "Test Loss":
        epoch_test_loss,

    "Training Time (s)":
        epoch_training_time
})


print(
    epoch_results.to_string(
        index=False
    )
)



print("\n")
print("=" * 60)
print("HIDDEN NEURON RESULTS")
print("=" * 60)


hidden_results = pd.DataFrame({

    "Hidden Neurons":
        hidden_neuron_values,

    "Train Accuracy (%)":
        hidden_train_accuracy,

    "Test Accuracy (%)":
        hidden_test_accuracy,

    "Test Loss":
        hidden_test_loss,

    "Training Time (s)":
        hidden_training_time
})


print(
    hidden_results.to_string(
        index=False
    )
)


print("\nAll experiments completed!")