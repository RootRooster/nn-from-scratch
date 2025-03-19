from network_template import Network, load_data_cifar


def main():
    train_file = "./data/train_data.pckl"
    test_file = "./data/test_data.pckl"
    train_data, train_class, test_data, test_class = load_data_cifar(
        train_file, test_file
    )
    val_pct = 0.1
    val_size = int(train_data.shape[1] * val_pct)
    # divide train data into
    # validation data
    val_data = train_data[..., :val_size]
    val_class = train_class[..., :val_size]
    # actual train data
    train_data = train_data[..., val_size:]
    train_class = train_class[..., val_size:]
    # The Network takes as input a list of the numbers of neurons at each layer. The first layer has to match the
    # number of input attributes from the data, and the last layer has to match the number of output classes
    # The initial settings are not even close to the optimal network architecture, try increasing the number of layers
    # and neurons and see what happens.

    # np_array.shape returns number of rows
    # a.k.a. number of input nodes
    # the shape means the number of neurons on each layer
    # [input, 100, 100, 10] - there are 4 layers first is the input whatever val_size
    # second layer has 100 neurons, so does the third. The final layer has 10 neurons.
    # This is also the output layer.
    net = Network([train_data.shape[0], 100, 100, 10], optimizer="sgd")
    net.train(train_data, train_class, val_data, val_class, 20, 64, 0.01)
    net.eval_network(test_data, test_class)


if __name__ == "__main__":
    main()
