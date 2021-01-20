"""Problem 3 - Training on MNIST"""
import numpy as np
from mytorch.tensor import Tensor
from mytorch.nn.loss import Loss, CrossEntropyLoss
from mytorch.optim.sgd import SGD
from mytorch.nn.linear import Linear
from mytorch.nn.batchnorm import BatchNorm1d
from mytorch.nn.sequential import Sequential
from mytorch.nn.activations import ReLU
# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100


def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)

    Args:
        train_x (np.array): training data (55000, 784)
        train_y (np.array): training labels (55000,)
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    model = Sequential(Linear(784, 20), ReLU(), Linear(20, 10))
    optimizer = SGD(model.parameters(), lr=0.1)
    criterion = CrossEntropyLoss()

    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(model, optimizer, criterion, train_x, train_y, val_x, val_y)
    return val_accuracies


def batches(train_x, train_y):
    result_x, result_y = [], []
    for i in range(0, len(train_x), BATCH_SIZE):
        result_x.append(train_x[i:i+BATCH_SIZE])
        result_y.append(train_y[i:i+BATCH_SIZE])
    return np.array(result_x), np.array(result_y)


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    model.train()
    val_accuracies = []

    # TODO: Implement me! (Pseudocode on writeup)
    # print(train_x[0][0], train_y[0])
    for e in range(num_epochs):
        train_y = train_y[None]

        pre_shuffle_data = np.hstack((train_x, train_y.T))
        np.random.shuffle(pre_shuffle_data)
        train_y = pre_shuffle_data[:, -1]
        train_x = pre_shuffle_data[:, :-1]

        x_batches, target_batches = batches(train_x, train_y)
        assert(len(x_batches) == len(target_batches))
        for i in range(len(x_batches)):
            # print(x_batches[i][0][0])
            # print(target_batches[i][0])
            # assert(False)
            optimizer.zero_grad()
            output = model.forward(Tensor(x_batches[i]))
            # print(output.shape)
            # assert(False)
            # print(target_batches[i].shape)
            loss = criterion.forward(output, Tensor(target_batches[i]))
            print(loss)
            loss.backward()
            optimizer.step()

            if (i % 100 == 0):
                accuracy = validate(model, val_x, val_y)
                print(accuracy)
                val_accuracies.append(accuracy)
                model.train()

    return val_accuracies


def get_labels(output):
    labels = []
    for vec in output.data:
        labels.append(np.argmax(vec))
    return np.array(labels)


def check_correct(predicted, target):
    assert(len(predicted) == len(target))
    correct = 0
    for i in range(len(target)):
        if predicted[i] == target[i]:
            correct += 1
    return correct


def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    # TODO: implement validation based on pseudocode
    model.eval()
    x_batches, target_batches = batches(val_x, val_y)
    num_correct = 0
    total = len(val_x)
    assert(len(val_x) == len(val_y))
    assert(len(x_batches) == len(target_batches))
    for (i) in range(len(x_batches)):
        output = model.forward(Tensor(x_batches[i]))
        labels = get_labels(output)
        num_correct += check_correct(labels, target_batches[i])

    return 100 * num_correct / total
