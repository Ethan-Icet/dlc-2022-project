"""
Training utilities.
"""

import torch
from torchvision import datasets
import time


def mnist_to_pairs(nb, input, target):
    input = torch.functional.F.avg_pool2d(input, kernel_size=2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes


def generate_pair_sets(nb):
    data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train=True, download=True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train=False, download=True)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    # train_input, train_target, train_classes, test_input, test_target, test_classes
    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)


def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros((target.size(0), target.max() + 1))
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp


def train_model(model, train_input, train_target, train_classes,
                mini_batch_size, criterion_class, criterion_leq, optimizer, nb_epochs=25,
                weight_loss_classes=0.0, weight_loss_pairs=1.0,
                freeze_epochs=0,
                one_hot_classes=True, one_hot_ioe=False):
    """
    Train a model on the MNIST dataset.
    :param model: The model to train.
    :param train_input: The training input.
    :param train_target: The training target.
    :param train_classes: The training classes.
    :param mini_batch_size: The mini-batch size.
    :param criterion: The loss function.
    :param optimizer: The optimizer.
    :param nb_epochs: The number of epochs.
    :param weight_loss_classes: The weight of the classification loss.
    :param weight_loss_pairs: The weight of the pairs loss.
    :param freeze_epochs: The number of epochs to freeze the model.
    :param one_hot_classes: Whether to convert the classes to one-hot labels.
    :param one_hot_ioe: Whether to convert the input and output to one-hot labels.
    :return: The trained model.
    """

    train_input, train_target = train_input.to(model.device), train_target.to(model.device)
    model.train()
    sum_loss = None
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            pred_ioe, (pred_label1, pred_label2) = model(train_input.narrow(0, b, mini_batch_size))
            loss_pairs = criterion(pred_ioe, train_target.narrow(0, b, mini_batch_size))
            loss_class1 = criterion(pred_label1, train_classes.narrow(0, b, mini_batch_size)[:, 0])
            loss_class2 = criterion(pred_label2, train_classes.narrow(0, b, mini_batch_size)[:, 1])
            loss = (weight_loss_pairs if freeze_epochs > nb_epochs else 0) * loss_pairs + \
                   weight_loss_classes * (loss_class1 + loss_class2)
            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {e}, loss {sum_loss}')

    model.eval()
    pred_ioe, (pred_label1, pred_label2) = model(train_input)
    if one_hot_ioe:
        _, predicted_ioe = torch.max(pred_ioe, 1)
    else:
        predicted_ioe = (pred_ioe > 0.5).long()
    if one_hot_classes:
        _, predicted_label1 = torch.max(pred_label1, 1)
        _, predicted_label2 = torch.max(pred_label2, 1)
    else:
        predicted_label1 = pred_label1
        predicted_label2 = pred_label2
    train_acc = (predicted_ioe == train_target).float().mean().item()
    train_acc_classes = ((predicted_label1 == train_classes[:, 0]) & (
                predicted_label2 == train_classes[:, 1])).float().mean().item()
    return sum_loss, train_acc, train_acc_classes


def evaluate_model(model,
                   mini_batch_size, criterion, optimizer, nb_epochs=25,
                   weight_loss_classes=0.0, weight_loss_pairs=1.0,
                   freeze_epochs=0,
                   convert_classes_to_one_hot=True, convert_ioe_to_one_hot=False):
    """
    Do 10 training cycles, print the time and return the losses and accuracies.
    The model has to be reset before each training cycle.
    :param model: the model to train
    :param mini_batch_size: the mini-batch size
    :param criterion: the loss function
    :param optimizer: the optimizer
    :param nb_epochs: the number of epochs
    :param weight_loss_classes: the weight of the classification loss
    :param weight_loss_pairs: the weight of the pairs loss
    :param freeze_epochs: the number of epochs to freeze the model
    :param convert_classes_to_one_hot: whether to convert the classes to one-hot labels
    :param convert_ioe_to_one_hot: whether to convert the input and output to one-hot labels
    :return: the losses and accuracies
    """

    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
    train_input, train_target = train_input.to(model.device), train_target.to(model.device)
    test_input, test_target = test_input.to(model.device), test_target.to(model.device)
    if convert_classes_to_one_hot:
        train_classes = convert_to_one_hot_labels(train_input, train_classes)
        test_classes = convert_to_one_hot_labels(test_input, test_classes)
    if convert_ioe_to_one_hot:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)
    train_losses = []
    train_accs = []
    train_accs_classes = []
    test_losses = []
    test_accs = []
    test_accs_classes = []
    for i in range(10):
        print(f'Cycle {i}')
        # Reset the model
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        start = time.perf_counter()
        train_loss, train_acc, train_acc_classes = train_model(model, train_input, train_target, train_classes,
                                                              mini_batch_size, criterion, optimizer, nb_epochs,
                                                              weight_loss_classes, weight_loss_pairs, freeze_epochs,
                                                              convert_classes_to_one_hot, convert_ioe_to_one_hot)
        end = time.perf_counter()
        print(f'Training time: {end - start}')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_accs_classes.append(train_acc_classes)
        model.eval()
        pred_ioe, (pred_label1, pred_label2) = model(test_input)
        if convert_ioe_to_one_hot:
            _, predicted_ioe = torch.max(pred_ioe, 1)
        else:
            predicted_ioe = (pred_ioe > 0.5).long()
        if convert_classes_to_one_hot:
            _, predicted_label1 = torch.max(pred_label1, 1)
            _, predicted_label2 = torch.max(pred_label2, 1)
        else:
            predicted_label1 = pred_label1
            predicted_label2 = pred_label2
        test_acc = (predicted_ioe == test_target).float().mean().item()
        test_acc_classes = ((predicted_label1 == test_classes[:, 0]) & (
                    predicted_label2 == test_classes[:, 1])).float().mean().item()
        test_accs.append(test_acc)
        test_accs_classes.append(test_acc_classes)
        print(f'Test accuracy: {test_acc}, test accuracy classes: {test_acc_classes}')
    return train_losses, train_accs, train_accs_classes, test_losses, test_accs, test_accs_classes

