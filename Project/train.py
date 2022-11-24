"""
Training utilities.
"""

import torch
from torch import nn
from torch.nn import functional as F
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


def output_to_predictions(out_leq: torch.Tensor, out_class1: torch.Tensor, out_class2: torch.Tensor,
                          one_hot_classes: bool = True, one_hot_leq: bool = False) \
        -> (torch.Tensor, (torch.Tensor, torch.Tensor)):
    """
    Converts the output of a model to predictions.

    Parameters
    ----------
    out_leq : torch.Tensor
        The output of the model for the less than or equal predictions.
    out_class1 : torch.Tensor
        The output of the model for the first class predictions
    out_class2 : torch.Tensor
        The output of the model for the second class predictions
    one_hot_classes : bool
        Whether the label predictions are one-hot vectors.
        If so it will be converted to a single integer.
        Otherwise, it will be returned as is.
    one_hot_leq : bool
        Whether to return the predictions are one-hot vectors.
        If so it will be converted to a binary value.
        Otherwise, we will check if it's above or below 0.5 and return a binary value.

    Returns
    -------
    predictions : torch.Tensor
        The predictions for the less than or equal predictions.
    classes : (torch.Tensor, torch.Tensor)
        The predictions for the class predictions.
    """
    if one_hot_classes and out_class1 is not None and out_class2 is not None:
        out_class1 = out_class1.argmax(1)
        out_class2 = out_class2.argmax(1)
    if one_hot_leq:
        out_leq = out_leq.argmax(1)
    else:
        out_leq = (out_leq > 0.5).long()
    return out_leq, (out_class1, out_class2)


def compute_predictions(model: nn.Module, data_input: torch.Tensor,
                        one_hot_classes: bool = True, one_hot_leq: bool = False):
    """
    Returns the predictions of a model for a given input.

    Parameters
    ----------
    model : nn.Module
        The model to use for the predictions.
    data_input : torch.Tensor
        The input to use for the predictions.
    one_hot_classes : bool
        Whether the label predictions are one-hot vectors.
        If so it will be converted to a single integer.
        Otherwise, it will be returned as is.
    one_hot_leq : bool
        Whether to return the predictions are one-hot vectors.
        If so it will be converted to a binary value.
        Otherwise, we will check if it's above or below 0.5 and return a binary value.

    Returns
    -------
    pred_leq: torch.Tensor[long] of shape (data_input.shape[0],)
        The less than or equal predictions for the given input.
    (pred_class1, pred_class2): (torch.Tensor[long], torch.Tensor[long])
        The labels predictions for the given input.
    """
    model.eval()
    with torch.no_grad():
        out_leq, (out_class1, out_class2) = model(data_input)
        return output_to_predictions(out_leq, out_class1, out_class2, one_hot_classes, one_hot_leq)


def compute_accuracy(model: nn.Module,
                     data_input: torch.Tensor, data_target: torch.Tensor, data_classes: torch.Tensor,
                     one_hot_classes: bool = True, one_hot_leq: bool = False) -> dict:
    """
    Computes the accuracy of a model for a given input and target.

    Parameters
    ----------
    model : nn.Module
        The model to use for the predictions.
    data_input : torch.Tensor
        The input to use for the predictions.
    data_target : torch.Tensor
        The target to use for the predictions.
    data_classes : torch.Tensor
        The classes to use for the predictions.
    one_hot_classes : bool
        Whether the label predictions are one-hot vectors.
        If so it will be converted to a single integer.
        Otherwise, it will be returned as is.
    one_hot_leq : bool
        Whether to return the predictions are one-hot vectors.
        If so it will be converted to a binary value.
        Otherwise, we will check if it's above or below 0.5 and return a binary value.

    Returns
    -------
    info : dict
        A dictionary containing the accuracy information.
    """
    pred_leq, (pred_class1, pred_class2) = compute_predictions(model, data_input,
                                                               one_hot_classes=one_hot_classes, one_hot_leq=one_hot_leq)
    acc_leq = (pred_leq == data_target).float().mean().item()
    acc_classes = acc_naive = None
    if pred_class1 is not None and pred_class2 is not None:
        acc_classes = ((pred_class1 == data_classes[:, 0]) & (pred_class2 == data_classes[:, 1])).float().mean().item()
        acc_naive = ((pred_class1 <= pred_class2) == data_target).float().mean().item()
    return {'acc_leq': acc_leq, 'acc_classes': acc_classes, 'acc_naive': acc_naive}


def train_model(model: nn.Module,
                train_input: torch.Tensor, train_target: torch.Tensor, train_classes: torch.Tensor,
                nb_epochs: int = 25, mini_batch_size: int = 100,
                criterion_classes=nn.MSELoss(), criterion_leq=nn.MSELoss(),
                optimizer: torch.optim = None,
                weight_loss_classes: float = 0.0, weight_loss_pairs: float = 1.0,
                freeze_epochs: int = 0,
                one_hot_classes: bool = True, one_hot_leq: bool = False,
                verbose: bool = False,
                test_input: torch.Tensor = None, test_target: torch.Tensor = None, test_classes: torch.Tensor = None,
                device=torch.device('cpu')) \
        -> dict:
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
    criterion_classes : nn.modules.loss
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
    verbose : bool
        Whether to print the loss at each epoch.
    test_input : torch.Tensor of shape (nb_samples, 2, 14, 14)
        The pair of MNIST digits to use for the test.
    test_target : torch.Tensor of shape (nb_samples, 1)
        The less or equal prediction to use for the test.
    test_classes : torch.Tensor of shape (nb_samples, 2, 10)
        The one-hot encoded labels to use for the test.

    Returns
    -------
    info : dict
        A dictionary containing the loss and accuracy at each epoch.
    """

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    eval_test = test_input is not None and test_target is not None and test_classes is not None

    info = {'train': {'loss': [], 'acc_leq': [], 'acc_classes': [], 'acc_naive': []},
            'test': {'loss': [], 'acc_leq': [], 'acc_classes': [], 'acc_naive': []}
            }

    data_input = train_input
    if one_hot_classes:
        data_classes = F.one_hot(train_classes, num_classes=10).float()
    else:
        data_classes = train_classes
    if one_hot_leq:
        data_target = F.one_hot(train_target, num_classes=2).float()
    else:
        data_target = train_target.float()

    data_input = data_input.to(device)
    data_classes = data_classes.to(device)
    data_target = data_target.to(device)

    if eval_test:
        test_data_input = test_input
        if one_hot_classes:
            test_data_classes = F.one_hot(test_classes, num_classes=10).float()
        else:
            test_data_classes = test_classes
        if one_hot_leq:
            test_data_target = F.one_hot(test_target, num_classes=2).float()
        else:
            test_data_target = test_target.float()

        test_data_input = test_data_input.to(device)
        test_data_classes = test_data_classes.to(device)
        test_data_target = test_data_target.to(device)

    for e in range(nb_epochs):
        train_loss = 0
        model.train()
        for b in range(0, train_input.size(0), mini_batch_size):
            pred_leq, (pred_label1, pred_label2) = model(data_input.narrow(0, b, mini_batch_size))

            loss_pairs = criterion_leq(pred_leq, data_target.narrow(0, b, mini_batch_size))
            if pred_label1 is not None:
                loss_class1 = criterion_classes(pred_label1, data_classes.narrow(0, b, mini_batch_size)[:, 0])
            else:
                loss_class1 = 0
            if pred_label2 is not None:
                loss_class2 = criterion_classes(pred_label2, data_classes.narrow(0, b, mini_batch_size)[:, 1])
            else:
                loss_class2 = 0

            if freeze_epochs < nb_epochs:
                loss = loss_class1 + loss_class2
            else:
                loss = weight_loss_pairs * loss_pairs + weight_loss_classes * (loss_class1 + loss_class2)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute and add the infos
        info['train']['loss'].append(train_loss)
        train_info = compute_accuracy(model, train_input, train_target, train_classes,
                                      one_hot_classes=one_hot_classes, one_hot_leq=one_hot_leq)
        info['train']['acc_leq'].append(train_info['acc_leq'])
        info['train']['acc_classes'].append(train_info['acc_classes'])
        info['train']['acc_naive'].append(train_info['acc_naive'])

        if eval_test:
            test_loss = 0
            model.eval()
            with torch.no_grad():
                for b in range(0, train_input.size(0), mini_batch_size):
                    pred_leq, (pred_label1, pred_label2) = model(test_data_input.narrow(0, b, mini_batch_size))

                    loss_pairs = criterion_leq(pred_leq, test_data_target.narrow(0, b, mini_batch_size))
                    if pred_label1 is not None:
                        loss_class1 = criterion_classes(pred_label1,
                                                        test_data_classes.narrow(0, b, mini_batch_size)[:, 0])
                    else:
                        loss_class1 = 0
                    if pred_label2 is not None:
                        loss_class2 = criterion_classes(pred_label2,
                                                        test_data_classes.narrow(0, b, mini_batch_size)[:, 1])
                    else:
                        loss_class2 = 0

                    if freeze_epochs < nb_epochs:
                        loss = loss_class1 + loss_class2
                    else:
                        loss = weight_loss_pairs * loss_pairs + weight_loss_classes * (loss_class1 + loss_class2)

                    test_loss += loss.item()

            info['test']['loss'].append(test_loss)
            test_info = compute_accuracy(model, test_input, test_target, test_classes,
                                         one_hot_classes=one_hot_classes, one_hot_leq=one_hot_leq)
            info['test']['acc_leq'].append(test_info['acc_leq'])
            info['test']['acc_classes'].append(test_info['acc_classes'])
            info['test']['acc_naive'].append(test_info['acc_naive'])

        if verbose:
            print(f'Epoch {e + 1}/{nb_epochs} - Train loss: {train_loss:.4f}')
            if eval_test:
                print(f'Epoch {e + 1}/{nb_epochs} - Test loss: {test_loss:.4f}')

    return info


def evaluate_model(model_builder, model_params,
                   nb_epochs=25, mini_batch_size=100,
                   criterion_classes=nn.MSELoss(), criterion_leq=nn.MSELoss(),
                   optimizer_builder=torch.optim.Adam, optimizer_params: dict = None,
                   weight_loss_classes: float = 0.0, weight_loss_pairs: float = 1.0,
                   freeze_epochs: int = 0,
                   one_hot_classes: bool = True, one_hot_leq: bool = False,
                   device=torch.device('cpu'), export_path: str = None) \
        -> dict:
    """
    Do 10 training cycles, print the time and return the losses and accuracies.
    The model has to be reset before each training cycle.

    Parameters
    ----------
    model_builder: function
        Function that returns a model.
    model_params: dict
        Parameters of the model.
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
    one_hot_classes : bool
        Whether the model outputs one-hot encoded labels.
    one_hot_leq : bool
        Whether the model outputs one-hot encoded less or equal prediction.
    device : torch.device
        The device to use.
    export_path : str
        The path to export the models.

    Returns
    -------
    info : dict
        The losses and accuracies.
    """

    if optimizer_params is None:
        optimizer_params = {}

    info = {'train': {'loss': [], 'acc_leq': [], 'acc_class1': [], 'acc_class2': []},
            'test': {'loss': [], 'acc_leq': [], 'acc_class1': [], 'acc_class2': []}}

    for i in range(10):
        print(f'Cycle {i + 1}/10')

        train_input, train_target, train_classes, test_input, test_target, test_classes = \
            generate_pair_sets(1000)

        # Create the model
        model = model_builder(**model_params)
        model.to(device)

        start = time.perf_counter()
        info_cycle = train_model(model,
                                 train_input, train_target, train_classes,
                                 nb_epochs=nb_epochs, mini_batch_size=mini_batch_size,
                                 criterion_classes=criterion_classes, criterion_leq=criterion_leq,
                                 optimizer=optimizer_builder(model.parameters(), **optimizer_params),
                                 weight_loss_classes=weight_loss_classes, weight_loss_pairs=weight_loss_pairs,
                                 freeze_epochs=freeze_epochs,
                                 one_hot_classes=one_hot_classes, one_hot_leq=one_hot_leq,
                                 verbose=True,
                                 test_input=test_input, test_target=test_target, test_classes=test_classes,
                                 device=device)
        end = time.perf_counter()
        print(f'Time: {end - start:.2f}s')

        for mode in info_cycle:
            for metric in info_cycle[mode]:
                info[mode][metric].append(info_cycle[mode][metric])

        if export_path is not None:
            # torch.save(model.state_dict(), export_path + f'weights_{i + 1}.pt')  # Export weights
            torch.save(model, export_path + f'model_{i + 1}.pt')  # Export weights and architecture

    return info


if __name__ == '__main__':
    # train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
    # print(train_classes.shape)
    # print(train_classes.argmax(dim=1))
    pass
