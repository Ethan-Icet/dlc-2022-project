import torch
import torch.nn as nn

from activation import ReLU, Tanh, Sigmoid, MSELoss
from linear import Linear
from sequential import Sequential, Module
from optimizer import SGD, Optimizer

import matplotlib.pyplot as plt


def generate_points(n=1000):
    """
    Generates a training and a test set of n points sampled uniformly in [0, 1]^2,
    each with a label 0 if outside the disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi), and 1 inside

    Parameters
    ----------
    n : int
        number of points to generate

    Returns
    -------
    train_input : torch.Tensor (n, 2)
        training set of n points
    train_target : torch.Tensor (n, 1)
        training set of n labels
    test_input : torch.Tensor (n, 2)
        test set of n points
    test_target : torch.Tensor (n, 1)
        test set of n labels
    """
    # Generate n points
    train_input = torch.rand(n, 2)
    test_input = torch.rand(n, 2)

    # Generate labels
    train_target = torch.zeros(n, 1, dtype=torch.long)
    test_target = torch.zeros(n, 1, dtype=torch.long)

    # Compute distance from center
    train_dist = torch.norm(train_input - 0.5, dim=1)
    test_dist = torch.norm(test_input - 0.5, dim=1)

    # Compute radius
    radius = 1 / torch.sqrt(torch.tensor(2 * torch.pi))

    # Assign labels
    train_target[train_dist <= radius] = 1
    test_target[test_dist <= radius] = 1

    return train_input, train_target, test_input, test_target


def generate_with_ratio(n=1000, ratio=0.5):
    """
    Similar to generate_points, but with a ratio of points inside the disk

    Parameters
    ----------
    n : int
        number of points to generate
    ratio : float
        ratio of points inside the disk

    Returns
    -------
    train_input : torch.Tensor (n, 2)
        training set of n points
    train_target : torch.Tensor (n, 1)
        training set of n labels
    test_input : torch.Tensor (n, 2)
        test set of n points
    test_target : torch.Tensor (n, 1)
        test set of n labels
    """
    # Generate n points
    train_input = torch.empty(n, 2, dtype=torch.float)
    test_input = torch.empty(n, 2, dtype=torch.float)

    # Generate labels
    train_target = torch.zeros(n, 1, dtype=torch.long)
    train_target[:int(n * ratio)] = 1
    test_target = torch.zeros(n, 1, dtype=torch.long)
    test_target[:int(n * ratio)] = 1

    # Compute radius
    radius = 1 / torch.sqrt(torch.tensor(2 * torch.pi))

    # Generate 10n points and keep only n of them (if not enough points for each label, rerun for more until enough)
    while True:
        # Generate 10n points
        points = torch.rand(10 * n, 2)

        # Compute distance from center
        dist = torch.norm(points - 0.5, dim=1)

        n_ones = (dist <= radius).sum()

        if n_ones >= 2 * n * ratio and 10 * n - n_ones >= 2 * n * (1 - ratio):
            break

    # Assign labels
    train_input[(train_target == 1)[:, 0]] = points[dist <= radius][:int(n * ratio)]
    train_input[(train_target == 0)[:, 0]] = points[dist > radius][:int(n * (1 - ratio))]
    test_input[(test_target == 1)[:, 0]] = points[dist <= radius][int(n * ratio):int(n * ratio) + int(n * ratio)]
    test_input[(test_target == 0)[:, 0]] = points[dist > radius][
                                           int(n * (1 - ratio)):int(n * (1 - ratio)) + int(n * (1 - ratio))]

    # Use pytorch to shuffle the data
    idx = torch.randperm(n)
    train_input = train_input[idx]
    train_target = train_target[idx]
    idx = torch.randperm(n)
    test_input = test_input[idx]
    test_target = test_target[idx]

    return train_input, train_target, test_input, test_target


def compute_accuracy_regular(model: nn.Module, data_input: torch.Tensor, data_target: torch.Tensor,
                             threshold: float = 0.5) -> float:
    """
    Computes the accuracy of the model on the given input and target

    Parameters
    ----------
    model : nn.Module
        model to test
    data_input : torch.Tensor
        input data
    data_target : torch.Tensor
        target data
    threshold : float
        threshold to decide if the output is 0 or 1

    Returns
    -------
    accuracy : float
        accuracy of the model on the given input and target
    """
    # Compute output
    output = model(data_input)

    # Compute accuracy
    accuracy = ((output > threshold).long() == data_target).float().mean().item()

    return accuracy


def compute_accuracy(model: Module, data_input: torch.Tensor, data_target: torch.Tensor,
                     threshold: float = 0.5) -> float:
    """
    Computes the accuracy of the model on the given input and target

    Parameters
    ----------
    model : Module
        model to test
    data_input : torch.Tensor
        input data
    data_target : torch.Tensor
        target data
    threshold : float
        threshold to decide if the output is 0 or 1

    Returns
    -------
    accuracy : float
        accuracy of the model on the given input and target
    """
    # Compute output
    output = model.forward(data_input)

    # Compute accuracy
    accuracy = ((output > threshold).long() == data_target).float().mean().item()

    return accuracy


def train_regular_model(model: nn.Module,
                        train_input: torch.Tensor, train_target: torch.Tensor,
                        test_input: torch.Tensor = None, test_target: torch.Tensor = None,
                        nb_epochs: int = 25, mini_batch_size: int = 100,
                        criterion=nn.MSELoss(),
                        optimizer: torch.optim = None,
                        verbose: bool = False) \
        -> dict:
    """
    Train a regular torch model on the dataset

    Parameters
    ----------
    model : Module
        model to train
    train_input : torch.Tensor (n, 2)
        training set of n points
    train_target : torch.Tensor (n, 1)
        training set of n labels
    test_input : torch.Tensor (n, 2)
        test set of n points
    test_target : torch.Tensor (n, 1)
        test set of n labels
    nb_epochs : int
        number of epochs to train for
    mini_batch_size : int
        size of mini-batches
    criterion : Module
        loss function
    optimizer : torch.optim
        optimizer
    verbose : bool
        whether to print loss at each epoch

    Returns
    -------
    dict
        dictionary containing the following keys:
        - train_loss: list of train losses
        - test_loss: list of test losses
        - train_error: list of train errors
        - test_error: list of test errors
    """

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    eval_test = test_input is not None and test_target is not None

    info = {'train_loss': [], 'test_loss': [], 'train_error': [], 'test_error': []}

    for e in range(nb_epochs):
        train_loss = 0

        model.train()
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        info['train_loss'].append(train_loss)
        info['train_error'].append(compute_accuracy_regular(model, train_input, train_target))

        if eval_test:
            test_loss = 0

            model.eval()
            for b in range(0, test_input.size(0), mini_batch_size):
                output = model(test_input.narrow(0, b, mini_batch_size))
                loss = criterion(output, test_target.narrow(0, b, mini_batch_size))

                test_loss += loss.item()

            info['test_loss'].append(test_loss)
            info['test_error'].append(compute_accuracy_regular(model, test_input, test_target))

        if verbose:
            print(
                f'Epoch {e + 1}/{nb_epochs}: '
                f'Train loss = {info["train_loss"][-1]:.4f}, Train error = {info["train_error"][-1]:.4f}')
            if eval_test:
                print(
                    f'Epoch {e + 1}/{nb_epochs}: '
                    f'Test loss = {info["test_loss"][-1]:.4f}, Test error = {info["test_error"][-1]:.4f}')

    return info


def train_model(model: Module,
                train_input: torch.Tensor, train_target: torch.Tensor,
                test_input: torch.Tensor = None, test_target: torch.Tensor = None,
                nb_epochs: int = 25, mini_batch_size: int = 100,
                criterion=MSELoss(),
                optimizer: Optimizer = None,
                verbose: bool = False) \
        -> dict:
    """
    Train a model created with our framework on the dataset

    Parameters
    ----------
    model : Module
        model to train
    train_input : torch.Tensor (n, 2)
        training set of n points
    train_target : torch.Tensor (n, 1)
        training set of n labels
    test_input : torch.Tensor (n, 2)
        test set of n points
    test_target : torch.Tensor (n, 1)
        test set of n labels
    nb_epochs : int
        number of epochs to train for
    mini_batch_size : int
        size of mini-batches
    criterion : Module
        loss function
    optimizer : torch.optim
        optimizer
    verbose : bool
        whether to print loss at each epoch

    Returns
    -------
    dict
        dictionary containing the following keys:
        - train_loss: list of train losses
        - test_loss: list of test losses
        - train_error: list of train errors
        - test_error: list of test errors
    """

    if optimizer is None:
        optimizer = SGD(model.param(), lr=0.01)
    eval_test = test_input is not None and test_target is not None

    info = {'train_loss': [], 'test_loss': [], 'train_error': [], 'test_error': []}

    for e in range(nb_epochs):
        train_loss = 0

        # model.train()
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))

            train_loss += loss.item()
            
            # Reset the gradient
            optimizer.zero_grad()
            # Compute the gradient
            grad = criterion.backward()
            model.backward(grad)
            # Update the parameters
            optimizer.step()

        info['train_loss'].append(train_loss)
        info['train_error'].append(compute_accuracy(model, train_input, train_target))

        if eval_test:
            test_loss = 0

            # model.eval()
            for b in range(0, test_input.size(0), mini_batch_size):
                output = model.forward(test_input.narrow(0, b, mini_batch_size))
                loss = criterion.forward(output, test_target.narrow(0, b, mini_batch_size))

                test_loss += loss.item()

            info['test_loss'].append(test_loss)
            info['test_error'].append(compute_accuracy(model, test_input, test_target))

        if verbose:
            print(
                f'Epoch {e + 1}/{nb_epochs}: '
                f'Train loss = {info["train_loss"][-1]:.4f}, Train error = {info["train_error"][-1]:.4f}')
            if eval_test:
                print(
                    f'Epoch {e + 1}/{nb_epochs}: '
                    f'Test loss = {info["test_loss"][-1]:.4f}, Test error = {info["test_error"][-1]:.4f}')

    return info


def plot_info(info: dict):
    """
    Plot the loss and error curves

    Parameters
    ----------
    info : dict
        dictionary containing the following keys:
        - train_loss: list of train losses
        - test_loss: list of test losses
        - train_error: list of train errors
        - test_error: list of test errors
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(info['train_loss'], label='Train')
    ax1.plot(info['test_loss'], label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(info['train_error'], label='Train')
    ax2.plot(info['test_error'], label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.legend()

    plt.show()


def plot_regions(model: Module, train_input: torch.Tensor, train_target: torch.Tensor,
                 test_input: torch.Tensor = None, test_target: torch.Tensor = None):
    """
    Plot the dataset with colored dots and the decision boundary

    Parameters
    ----------
    model : Module
        trained model
    train_input : torch.Tensor (n, 2)
        training set of n points
    train_target : torch.Tensor (n, 1)
        training set of n labels
    test_input : torch.Tensor (n, 2)
        test set of n points
    test_target : torch.Tensor (n, 1)
        test set of n labels
    """

    plt.figure(figsize=(10, 10))

    # Plot the decision boundary
    n_points = 1000
    x = torch.linspace(0, 1, n_points)
    y = torch.linspace(0, 1, n_points)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    output = model.forward(grid).view(n_points, n_points).detach().numpy()
    plt.contourf(xx, yy, output, cmap='coolwarm', alpha=0.5)

    # Show color bar
    plt.colorbar()

    # Plot the training set
    plt.scatter(train_input[:, 0], train_input[:, 1], c=train_target[:, 0], s=5, cmap='coolwarm')

    # Plot the test set
    if test_input is not None and test_target is not None:
        plt.scatter(test_input[:, 0], test_input[:, 1], c=test_target[:, 0], s=5, cmap='coolwarm', marker='^')

        # Add legend
        labels = ['Train', 'Test']
        handles = [plt.Line2D([], [], color='black', marker='o', linestyle='None'),
                   plt.Line2D([], [], color='black', marker='^', linestyle='None')]
        plt.legend(handles, labels, loc='upper right')

    # Make the axis square
    plt.axis('square')

    plt.show()


def main():
    train_input, train_target, test_input, test_target = generate_with_ratio(1000)

    # Print shape and type
    print('Train input:', train_input.shape, train_input.dtype)
    print('Train target:', train_target.shape, train_target.dtype)
    print('Test input:', test_input.shape, test_input.dtype)
    print('Test target:', test_target.shape, test_target.dtype)
    # Make a model
    n = 25
    model = Sequential(Linear(2, n), Tanh(), Linear(n, n), ReLU(), Linear(n, 1), Sigmoid())
    print(model)
    # Train the model
    n_epochs = 100
    mini_batch_size = 10
    criterion = MSELoss()
    optimizer = SGD(model.param(), lr=0.01)
    info = train_model(model, train_input, train_target, test_input, test_target,
                       nb_epochs=n_epochs, mini_batch_size=mini_batch_size,
                       criterion=criterion, optimizer=optimizer, verbose=True)
    # Test the model
    accuracy = compute_accuracy(model, test_input, test_target)
    print(f'Accuracy: {accuracy:.4f}')

    # Plot the loss and error curves
    plot_info(info)

    # Plot the decision boundary
    plot_regions(model, train_input, train_target, test_input, test_target)


if __name__ == '__main__':
    main()
