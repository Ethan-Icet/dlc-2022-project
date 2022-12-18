#!/usr/bin/python3

from train import *

"""
You must implement a test executable that imports the framework and
• Generates a training and a test set of 1000 points sampled uniformly in [0, 1]^2, each with a
  label 0 if outside the disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi), and 1 inside,
• builds a network with two input units, one output unit, three hidden layers of 25 units,
• trains it with MSE, logging the loss,
• computes and prints the final train and the test errors.
"""

if __name__ == '__main__':

    train_input, train_target, test_input, test_target = generate_with_ratio(1000, ratio=0.5)

    # Print shape and type
    print('Train input:', train_input.shape, train_input.dtype)
    print('Train target:', train_target.shape, train_target.dtype)
    print('Test input:', test_input.shape, test_input.dtype)
    print('Test target:', test_target.shape, test_target.dtype)
    # Make a model
    n = 25
    model = Sequential(Linear(2, n), Tanh(), Linear(n, n), ReLU(), Linear(n, 1), Sigmoid())
    print('Model:', model, sep='\n')

    # Train the model
    nb_epochs = 100
    mini_batch_size = 10
    criterion = MSELoss()
    optimizer = SGD(model.param(), lr=0.01)

    for e in range(nb_epochs):
        train_loss = 0

        # model.train()
        for b in range(0, train_input.size(0), mini_batch_size):
            mini_batch_input = train_input.narrow(0, b, mini_batch_size)
            mini_batch_target = train_target.narrow(0, b, mini_batch_size)

            output = model.forward(mini_batch_input)
            loss = criterion.forward(output, mini_batch_target)

            train_loss += loss.item()

            # Reset the gradient
            optimizer.zero_grad()
            # Compute the gradient
            grad = criterion.backward()
            model.backward(grad)
            # Update the parameters
            optimizer.step()

        # print(f'Epoch {e + 1}/{nb_epochs}: '
        #       f'loss = {train_loss:.4f}', end="\r", flush=True)
        print(f'Epoch {e + 1}/{nb_epochs}: loss = {train_loss:.4f}')

    # model.eval()

    # Compute and print the accuracies
    train_output = model.forward(train_input)
    test_output = model.forward(test_input)
    train_pred = (train_output > 0.5).long()
    test_pred = (test_output > 0.5).long()
    train_acc = (train_pred == train_target).float().mean()
    test_acc = (test_pred == test_target).float().mean()
    print(f'Train accuracy: {train_acc:.4f}')
    print(f'Test accuracy: {test_acc:.4f}')
