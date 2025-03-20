# import pandas as pd
import numpy as np
import pickle

LOGGING = True


class Network:
    def __init__(self, sizes, optimizer="sgd", l2_lambda=0.0, beta1=0.9, beta2=0.99):
        """
        Takes in an array of `sizes` which is an array of numbers. Each represent the number of neurons on each layer.

        Args:
            sizes (list): List of integers representing number of neurons in each layer
            optimizer (str): Optimization algorithm to use ("sgd" or "adam")
            l2_lambda (float): L2 regularization parameter (default: 0.0)
            beta1 (float): Adam beta1 parameter
            beta2 (float): Adam beta2 parameter

        Weights are initized as such that the row means n-1 th layer. Therefore if we want to which weights
        the first neuron of n-1 layer connects to we just look at the row 1. Aka first neuron.
        Same for colums - they represent the weights from which the neuron 'n' gets its values.
        """
        # weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        # the weights for that layer of dimensions size(L+1) X size(L)
        # the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        # in layer L is therefore size(L+1).
        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.weights = [
            ((2 / sizes[i - 1]) ** 0.5) * np.random.randn(sizes[i], sizes[i - 1])
            for i in range(1, len(sizes))
        ]
        self.biases = [
            np.zeros((x, 1)) for x in sizes[1:]
        ]  # biases are 1D arrays filled with zeros of length x
        self.optimizer = optimizer
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.l2_lambda = l2_lambda  # L2 regularization parameter
        if self.optimizer == "adam":
            # Initialize Adam parameters
            self.beta1 = beta1  # Exponential decay rate for first moment
            self.beta2 = beta2  # Exponential decay rate for second moment
            self.epsilon = 1e-8  # Small constant to prevent division by zero

            # Initialize first moment (momentum) for weights and biases
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]

            # Initialize second moment (RMSprop) for weights and biases
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.v_b = [np.zeros_like(b) for b in self.biases]

            # Initialize timestep
            self.t = 0

    def train(
        self,
        training_data,
        training_class,
        val_data,
        val_class,
        epochs,
        mini_batch_size,
        eta,
        decay_rate=0.0,  # k in the formula nt = n * e^(-kt)
    ):
        """
        training data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        n0 is the number of input attributes
        training_class - numpy array of dimensions [c x m], where c is the number of classes
        epochs - number of passes over the dataset
        mini_batch_size - number of examples the network uses to compute the gradient estimation
        eta - initial learning rate
        decay_rate - exponential decay rate for learning rate (default: 0.0, no decay)
        """
        n = training_data.shape[1]
        for j in range(epochs):
            # Calculate decayed learning rate for this epoch
            # nt = n * e^(-kt) where:
            # n = initial learning rate (eta)
            # k = decay rate
            # t = current epoch
            eta_current = eta * np.exp(-decay_rate * j)

            if LOGGING:
                print(f"Epoch {j}, Learning Rate: {eta_current:.6f}")
            loss_avg = 0.0
            mini_batches = [
                {
                    "training_data": training_data[
                        :, k : k + mini_batch_size
                    ],  # takes n columns of the matrix training data - each column represents the input
                    "training_class": training_class[
                        :, k : k + mini_batch_size
                    ],  # same but for the output
                }
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                output, Zs, As = self.forward_pass(mini_batch["training_data"])
                gw, gb = self.backward_pass(
                    output, mini_batch["training_class"], Zs, As
                )

                self.update_network(gw, gb, eta_current)

                loss = cross_entropy(
                    mini_batch["training_class"],
                    output,
                    weights=self.weights if self.l2_lambda > 0 else None,
                    l2_lambda=self.l2_lambda,
                )
                loss_avg += loss

            if LOGGING:
                print("Epoch {} complete".format(j))
                print("Loss:" + str(loss_avg / len(mini_batches)))
            if j % 10 == 0:
                self.eval_network(val_data, val_class)

    def eval_network(self, validation_data, validation_class):
        """
        validation data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        n0 is the number of input attributes
        validation_class - numpy array of dimensions [c x m], where c is the number of classes

        returns (tuple) (float) validation_loss (float) classification_accuracy
        """
        n = validation_data.shape[1]
        loss_avg = 0.0
        tp = 0.0
        for i in range(validation_data.shape[1]):
            example = np.expand_dims(validation_data[:, i], -1)
            example_class = np.expand_dims(validation_class[:, i], -1)
            example_class_num = np.argmax(validation_class[:, i], axis=0)
            output, _, _ = self.forward_pass(example)
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(
                example_class,
                output,
                weights=self.weights if self.l2_lambda > 0 else None,
                l2_lambda=self.l2_lambda,
            )
            loss_avg += loss
        valtidation_loss = loss_avg / n
        classification_accuracy = tp / n
        if LOGGING:
            print("Validation Loss:" + str(valtidation_loss))
            print("Classification accuracy: " + str(classification_accuracy))
        return valtidation_loss, classification_accuracy

    def update_network(self, gw, gb, eta):
        """
        Update network parameters using the specified optimizer.

        Args:
            gw (list): Weight gradients - list with elements of the same shape as elements in self.weights
            gb (list): Bias gradients - list with elements of the same shape as elements in self.biases
            eta (float): Learning rate
        """
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= eta * gw[i]
                self.biases[i] -= eta * gb[i]
        elif self.optimizer == "adam":
            self.t += 1  # Increment timestep

            for i in range(len(self.weights)):
                # Update biased first moment estimate (momentum)
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gw[i]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gb[i]

                # Update biased second raw moment estimate (RMSprop)
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * np.square(
                    gw[i]
                )
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(
                    gb[i]
                )

                # Compute bias-corrected first and second moment estimates
                m_w_hat = self.m_w[i]  # / (1 - np.power(self.beta1, self.t))
                m_b_hat = self.m_b[i]  # / (1 - np.power(self.beta1, self.t))
                v_w_hat = self.v_w[i]  # / (1 - np.power(self.beta2, self.t))
                v_b_hat = self.v_b[i]  # / (1 - np.power(self.beta2, self.t))

                # Apply Adam updates
                self.weights[i] -= eta * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                self.biases[i] -= eta * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        else:
            raise ValueError("Unknown optimizer:" + self.optimizer)

    def forward_pass(self, input):
        """
        input - numpy array of dimensions [n0 x m], where m is the number of examples in the mini batch and
        n0 is the number of input attributes
        """
        activation = input
        activations = [activation]
        z = None
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # set the output special case with softmax - if there is no z just set output layer to zeros
        z = zs[-1]
        output = softmax(z)
        # remove the last activation layer since it should actually be the output layer
        activations.pop()
        return output, zs, activations

    def backward_pass(self, output, target, Zs, activations):
        """
        Function takes in:
          - `output` = matrix; its columns represent results in different training examples
          - `target` = matrix; columns represent desired results
          - `Zs` = array of matries; each element in the array represents a different layer in the NN
            Each layer is a matrix from multiple differnet examples. Each example is presented as a column
          - `activations` = array of matrices; each element in the array represents the output value of the neuron
            first element is just the initial value of the NN. Each example is presented by a column
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Compute output layer error (delta)
        delta = softmax_dLdZ(output, target)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-1].transpose())

        # Backpropagate the error
        for i in range(2, self.num_layers):
            z = Zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i].transpose())

        # Average the gradients over the batch
        nabla_b = [np.mean(matrix, axis=1, keepdims=True) for matrix in nabla_b]

        # Add L2 regularization gradient to weights
        if self.l2_lambda > 0:
            batch_size = output.shape[1]
            for i in range(len(nabla_w)):
                nabla_w[i] += (self.l2_lambda / batch_size) * self.weights[i]

        return nabla_w, nabla_b


def softmax(Z):
    """
    Calculates the softmax function for every column in the matrix `Z`
    """
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / expZ.sum(axis=0, keepdims=True)


def softmax_dLdZ(output, target):
    # partial derivative of the cross entropy loss w.r.t Z at the last layer
    return output - target


def cross_entropy(y_true, y_pred, epsilon=1e-12, weights=None, l2_lambda=0.0):
    """
    Compute cross entropy loss with optional L2 regularization.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        epsilon: Small constant for numerical stability
        weights: List of weight matrices for L2 regularization
        l2_lambda: L2 regularization parameter

    Returns:
        Total loss (cross entropy + L2 regularization if weights provided)
    """
    targets = y_true.transpose()
    predictions = y_pred.transpose()
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]

    # Calculate cross entropy loss
    ce = -np.sum(targets * np.log(predictions + epsilon)) / N

    # Add L2 regularization term if weights are provided
    if weights is not None and l2_lambda > 0:
        l2_loss = 0.0
        for w in weights:
            l2_loss += np.sum(np.square(w))
        ce += 0.5 * l2_lambda * l2_loss / N

    return ce


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


def load_data_cifar(train_file, test_file):
    """
    Expects the `train_file` and the `test_file` paths given as strings.
    Returns four matrices.
    1. train_data matrix (np.array) - rows represent input vectors.
    2. train_class_one_hot (np.array) - rows represent output vectors.
    3. test_data matrix (np.array) - rows represent input vectors.
    4. test_class_one_hot (np.array) - rows represent output vectors.
    """
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)
    train_data = np.array(train_dict["data"]) / 255.0  # normalize the data
    train_class = np.array(train_dict["labels"])
    train_class_one_hot = np.zeros((train_data.shape[0], 10))

    # this code creates a variable called train_class_one_hot that is first set to all zero. Matrix
    # of all elements of zero. Then for each row in the matrix look at what is the number of the train class
    # and then set that number to 1.0.
    # The np.arange creates a list of elements [0, 1, 2 ,3... shape[0] - 1]
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0
    test_data = np.array(test_dict["data"]) / 255.0  # normalize the data
    test_class = np.array(test_dict["labels"])
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0

    # we transpose the final solutions to get the colums that represent each of the examples instead of the rows
    return (
        train_data.transpose(),
        train_class_one_hot.transpose(),
        test_data.transpose(),
        test_class_one_hot.transpose(),
    )
