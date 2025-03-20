from network_template import Network, load_data_cifar
from datetime import datetime


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

    # task one:
    # optimizers = ["adam", "sgd"]
    # learning_rates = [0.001, 0.0001, 0.00001]  # 0.1, 0.2, 0.5, 0.8]
    # for optimizer in optimizers:
    #     print(f"{optimizer} optimizer tests: \n \n")
    #     for learning_rate in learning_rates:
    #         net = Network(
    #             sizes=[train_data.shape[0], 100, 100, 10], optimizer=optimizer
    #         )
    #         net.train(
    #             train_data,
    #             train_class,
    #             val_data,
    #             val_class,
    #             epochs=20,
    #             mini_batch_size=64,
    #             eta=learning_rate,
    #             # decay_rate=0.1,
    #         )
    #         valtidation_loss, classification_accuracy = net.eval_network(
    #             test_data, test_class
    #         )
    #         print(f"Learning rate: {learning_rate}")
    #         print("Validation Loss:" + str(valtidation_loss))
    #         print("Classification accuracy: " + str(classification_accuracy))
    #         print()

    # task 2
    # l2_lambdas = [0.001, 0.005, 0.01, 0.1]

    # for l2_lambda in l2_lambdas:
    #     net = Network(
    #         sizes=[train_data.shape[0], 100, 100, 10],
    #         optimizer="sgd",
    #         l2_lambda=l2_lambda,
    #     )

    #     print()
    #     print(f"For l2_lambda = {l2_lambda}")
    #     # task 2
    #     net.train(
    #         train_data,
    #         train_class,
    #         val_data,
    #         val_class,
    #         epochs=20,
    #         mini_batch_size=64,
    #         eta=0.005,
    #         # decay_rate=0.1,
    #     )
    #     _, _ = net.eval_network(test_data, test_class)

    # # 3rd final task
    # decay_rates = [0, 0.001, 0.005, 0.01, 0.1, 0.3, 0.5]
    # for decay_rate in decay_rates:
    #     net = Network(sizes=[train_data.shape[0], 100, 100, 10], optimizer="sgd")
    #     net.train(
    #         train_data,
    #         train_class,
    #         val_data,
    #         val_class,
    #         epochs=20,
    #         mini_batch_size=64,
    #         eta=0.005,
    #         decay_rate=decay_rate,
    #     )
    #     valtidation_loss, classification_accuracy = net.eval_network(
    #         test_data, test_class
    #     )

    #     if decay_rate == 0:
    #         print("Without decay rate:")
    #     else:
    #         print("Decday rate:" + str(decay_rate))
    #     print("Validation Loss:" + str(valtidation_loss))
    #     print("Classification accuracy: " + str(classification_accuracy))
    #     print()

    # New configuration with all parameters defined as variables
    # Network parameters
    network_params = {
        'sizes': [train_data.shape[0], 100, 100, 10],  # Default architecture
        'optimizer': "sgd",  # Default optimizer
        'l2_lambda': 0.0,  # Default L2 regularization
        'beta1': 0.9,  # Default Adam beta1
        'beta2': 0.99,  # Default Adam beta2
    }

    # Training parameters
    training_params = {
        'epochs': 20,  # Default epochs
        'mini_batch_size': 64,  # Default batch size
        'eta': 0.005,  # Default learning rate
        'decay_rate': 0.0,  # Default decay rate
    }

    # Create and train the network
    net = Network(**network_params)
    
    # Print network specifications
    print("\nNetwork Specifications:")
    print("-" * 50)
    for param, value in network_params.items():
        print(f"{param}: {value}")
    print("\nTraining Parameters:")
    print("-" * 50)
    for param, value in training_params.items():
        print(f"{param}: {value}")
    print("-" * 50)

    # Train the network
    net.train(
        train_data,
        train_class,
        val_data,
        val_class,
        **training_params
    )

    # Evaluate the network
    validation_loss, classification_accuracy = net.eval_network(test_data, test_class)

    # Print results
    print("\nResults:")
    print("-" * 50)
    print(f"Validation Loss: {validation_loss}")
    print(f"Classification Accuracy: {classification_accuracy}")
    print("-" * 50)

    # Write results to file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results.txt", "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Run Date: {timestamp}\n\n")
        
        f.write("Network Specifications:\n")
        f.write("-" * 50 + "\n")
        for param, value in network_params.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\nTraining Parameters:\n")
        f.write("-" * 50 + "\n")
        for param, value in training_params.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\nResults:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Validation Loss: {validation_loss}\n")
        f.write(f"Classification Accuracy: {classification_accuracy}\n")
        f.write("=" * 80 + "\n")


if __name__ == "__main__":
    main()
