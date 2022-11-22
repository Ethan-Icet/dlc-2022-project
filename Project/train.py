"""
Training utilities.
"""

import torch
from torch import nn
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


def train_model(model: nn.Module,
                train_input: torch.Tensor, train_target: torch.Tensor, train_classes: torch.Tensor,
                nb_epochs: int = 25, mini_batch_size: int = 100,
                criterion_class=nn.CrossEntropyLoss, criterion_leq=nn.CrossEntropyLoss,
                optimizer: torch.optim = None,
                weight_loss_classes: float = 0.0, weight_loss_pairs: float = 1.0,
                freeze_epochs: int = 0,
                one_hot_classes: bool = True, one_hot_leq: bool = False):
    """
    Train a model on a pair of digit from the MNIST dataset who outputs the less or equal prediction and the labels.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_input : torch.Tensor of shape (nb_samples, 2, 14, 14)
        The pair of MNIST digits.
    train_target : torch.Tensor of shape (nb_samples, 1)
        The less or equal prediction.
    train_classes : torch.Tensor of shape (nb_samples, 2, 10)
        The one-hot encoded labels.
    nb_epochs : int
        The number of epochs.
    mini_batch_size : int
        The mini-batch size.
    criterion_class : nn.modules.loss
        The loss function for the classes.
    criterion_leq : nn.modules.loss
        The loss function for the less or equal prediction.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    weight_loss_classes : float
        The weight of the loss for the classes.
    weight_loss_pairs : float
        The weight of the loss for the less or equal prediction.
    freeze_epochs : int
        The number of epochs to freeze the weights of the model for the less or equal prediction.
    one_hot_classes : bool
        Whether the model outputs one-hot encoded labels.
    one_hot_leq : bool
        Whether the model outputs one-hot encoded less or equal prediction.

    Returns
    -------
    sum_loss : float
        The sum of the training loss.
    train_acc_classes : float
        The training accuracy for the label prediction.
    train_acc_leq : float
        The training accuracy for the less or equal prediction.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    train_input, train_target = train_input.to(model.device), train_target.to(model.device)
    model.train()
    sum_loss = None
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            pred_leq, (pred_label1, pred_label2) = model(train_input.narrow(0, b, mini_batch_size))
            loss_pairs = criterion_leq(pred_leq, train_target.narrow(0, b, mini_batch_size))
            if pred_label1 is not None:
                loss_class1 = criterion_class(pred_label1, train_classes.narrow(0, b, mini_batch_size)[:, 0])
            else:
                loss_class1 = 0
            if pred_label2 is not None:
                loss_class2 = criterion_class(pred_label2, train_classes.narrow(0, b, mini_batch_size)[:, 1])
            else:
                loss_class2 = 0
            if freeze_epochs < nb_epochs:
                loss = loss_class1 + loss_class2
            else:
                loss = weight_loss_pairs * loss_pairs + weight_loss_classes * (loss_class1 + loss_class2)
            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {e}, loss {sum_loss}')

    model.eval()
    pred_leq, (pred_label1, pred_label2) = model(train_input)
    if one_hot_leq:
        _, predicted_leq = torch.max(pred_leq, 1)
    else:
        predicted_leq = (pred_leq > 0.5).long()
    if one_hot_classes:
        _, predicted_label1 = torch.max(pred_label1, 1)
        _, predicted_label2 = torch.max(pred_label2, 1)
    else:
        predicted_label1 = pred_label1
        predicted_label2 = pred_label2
    train_acc = (predicted_leq == train_target).float().mean().item()
    train_acc_classes = ((predicted_label1 == train_classes[:, 0]) & (
            predicted_label2 == train_classes[:, 1])).float().mean().item()
    return sum_loss, train_acc, train_acc_classes


def evaluate_model(model,
                   nb_epochs=25, mini_batch_size=100,
                   criterion_classes=nn.CrossEntropyLoss, criterion_leq=nn.CrossEntropyLoss,
                   optimizer_builder=torch.optim.Adam, optimizer_params: dict = None,
                   weight_loss_classes: float = 0.0, weight_loss_pairs: float = 1.0,
                   freeze_epochs: int = 0,
                   convert_classes_to_one_hot: bool = True, convert_leq_to_one_hot: bool = False):
    """
    Do 10 training cycles, print the time and return the losses and accuracies.
    The model has to be reset before each training cycle.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    nb_epochs : int
        The number of epochs.
    mini_batch_size : int
        The mini-batch size.
    criterion_classes : nn.modules.loss
        The loss function for the classes.
    criterion_leq : nn.modules.loss
        The loss function for the less or equal prediction.
    optimizer_builder : torch.optim.Optimizer
        The optimizer to use.
    optimizer_params : dict
        The parameters for the optimizer.
    weight_loss_classes : float
        The weight of the loss for the classes.
    weight_loss_pairs : float
        The weight of the loss for the less or equal prediction.
    freeze_epochs : int
        The number of epochs to freeze the weights of the model for the less or equal prediction.
    convert_classes_to_one_hot : bool
        Whether the model outputs one-hot encoded labels.
    convert_leq_to_one_hot : bool
        Whether the model outputs one-hot encoded less or equal prediction.

    Returns
    -------
    train_losses : list of float
        The training losses.
    train_accs_classes : list of float
        The training accuracies for the label prediction.
    train_accs_leq : list of float
        The training accuracies for the less or equal prediction.
    test_losses : list of float
        The test losses.
    test_accs_classes : list of float
        The test accuracies for the label prediction.
    test_accs_leq : list of float
        The test accuracies for the less or equal prediction.
    """

    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
    train_input, train_target = train_input.to(model.device), train_target.to(model.device)
    test_input, test_target = test_input.to(model.device), test_target.to(model.device)
    if convert_classes_to_one_hot:
        train_classes = convert_to_one_hot_labels(train_input, train_classes)
        test_classes = convert_to_one_hot_labels(test_input, test_classes)
    if convert_leq_to_one_hot:
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
                                                                nb_epochs, mini_batch_size,
                                                                criterion_classes, criterion_leq,
                                                                optimizer_builder(**optimizer_params),
                                                                weight_loss_classes, weight_loss_pairs,
                                                                freeze_epochs,
                                                                convert_classes_to_one_hot, convert_leq_to_one_hot)

        end = time.perf_counter()
        print(f'Training time: {end - start}')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_accs_classes.append(train_acc_classes)
        model.eval()
        pred_leq, (pred_label1, pred_label2) = model(test_input)
        if convert_leq_to_one_hot:
            _, predicted_leq = torch.max(pred_leq, 1)
        else:
            predicted_leq = (pred_leq > 0.5).long()
        if convert_classes_to_one_hot:
            _, predicted_label1 = torch.max(pred_label1, 1)
            _, predicted_label2 = torch.max(pred_label2, 1)
        else:
            predicted_label1 = pred_label1
            predicted_label2 = pred_label2
        test_acc = (predicted_leq == test_target).float().mean().item()
        test_acc_classes = ((predicted_label1 == test_classes[:, 0]) & (
                predicted_label2 == test_classes[:, 1])).float().mean().item()
        test_accs.append(test_acc)
        test_accs_classes.append(test_acc_classes)
        print(f'Test accuracy: {test_acc}, test accuracy classes: {test_acc_classes}')
    return train_losses, train_accs, train_accs_classes, test_losses, test_accs, test_accs_classes
