"""
Template models used for the experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, ch_in=1, ch1=64, ch2=64,
                 conv1_kernel=3, conv2_kernel=6,
                 use_max_pool1=True, max_pool1_kernel=2, max_pool1_stride=2,
                 use_max_pool2=False, max_pool2_kernel=2, max_pool2_stride=1,
                 dropout1=0.0, dropout2=0.0,
                 activation1=nn.ReLU(), activation2=nn.ReLU(),
                 use_batch_norm=True, use_skip_connections=False):
        super().__init__()
        self.use_max_pool1 = use_max_pool1
        self.use_max_pool2 = use_max_pool2
        self.use_batch_norm = use_batch_norm
        self.skip_connections = use_skip_connections

        self.conv1 = nn.Conv2d(ch_in, ch1, conv1_kernel)
        self.max_pool1 = nn.MaxPool2d(max_pool1_kernel, max_pool1_stride)
        self.batch_norm1 = nn.BatchNorm2d(ch1)
        self.activation1 = activation1
        self.dropout1 = nn.Dropout(dropout1)
        self.conv2 = nn.Conv2d(ch1, ch2, conv2_kernel)
        self.max_pool2 = nn.MaxPool2d(max_pool2_kernel, max_pool2_stride)
        self.batch_norm2 = nn.BatchNorm2d(ch2)
        self.activation2 = activation2
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        y = self.conv1(x)
        if self.use_max_pool1:
            y = self.max_pool1(y)
        if self.use_batch_norm:
            y = self.batch_norm1(y)
        y = self.activation1(y)
        if self.dropout1.p > 0:
            y = self.dropout1(y)

        y = self.conv2(y)
        if self.use_max_pool2:
            y = self.max_pool2(y)
        if self.use_batch_norm:
            y = self.batch_norm2(y)
        if self.skip_connections:
            y = y + x
        y = self.activation2(y)
        if self.dropout2.p > 0:
            y = self.dropout2(y)

        return y


class ConvBlockReBa(nn.Module):
    # Apply BatchNorm after ReLU
    # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
    def __init__(self, ch_in=1, ch1=64, ch2=64,
                 conv1_kernel=3, conv2_kernel=6,
                 use_max_pool1=True, max_pool1_kernel=3, max_pool1_stride=1,
                 use_max_pool2=True, max_pool2_kernel=2, max_pool2_stride=1,
                 dropout1=0.0, dropout2=0.0,
                 activation1=nn.ReLU(), activation2=nn.ReLU(),
                 use_skip_connections=False):
        super().__init__()
        self.use_max_pool1 = use_max_pool1
        self.use_max_pool2 = use_max_pool2
        self.skip_connections = use_skip_connections

        self.conv1 = nn.Conv2d(ch_in, ch1, conv1_kernel)
        self.max_pool1 = nn.MaxPool2d(max_pool1_kernel, max_pool1_stride)
        self.batch_norm1 = nn.BatchNorm2d(ch1)
        self.activation1 = activation1
        self.dropout1 = nn.Dropout(dropout1)
        self.conv2 = nn.Conv2d(ch1, ch2, conv2_kernel)
        self.max_pool2 = nn.MaxPool2d(max_pool2_kernel, max_pool2_stride)
        self.batch_norm2 = nn.BatchNorm2d(ch2)
        self.activation2 = activation2
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        y = self.conv1(x)
        if self.use_max_pool1:
            y = self.max_pool1(y)
        y = self.activation1(y)
        if self.dropout1.p > 0:
            y = self.dropout1(y)
        y = self.batch_norm1(y)

        y = self.conv2(y)
        if self.use_max_pool2:
            y = self.max_pool2(y)
        if self.skip_connections:
            y = y + x
        y = self.activation2(y)
        if self.dropout2.p > 0:
            y = self.dropout2(y)
        y = self.batch_norm2(y)

        return y


class FCBlock(nn.Module):
    def __init__(self, input_size=None, fc=64, out=10,
                 dropout=0.0,
                 activation1=nn.ReLU(), activation2=nn.ReLU(),
                 use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        self.fc1 = nn.Linear(
            input_size, fc) if input_size else nn.LazyLinear(fc)
        self.activation1 = activation1
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(fc)
        self.fc2 = nn.Linear(fc, out)
        self.activation2 = activation2

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation1(y)
        if self.dropout.p > 0:
            y = self.dropout(y)
        if self.use_batch_norm:
            y = self.batch_norm(y)
        y = self.fc2(y)
        y = self.activation2(y)
        return y


def build_predictFC():
    return nn.Sequential(
        nn.Linear(20, 1),
        nn.Sigmoid()
    )


def build_predictFC2():
    return nn.Sequential(
        nn.Linear(20, 2),
        nn.Softmax(dim=1)
    )


def predict_label_then_result(x):
    x1, x2 = x[..., :10], x[..., 10:]
    y1 = torch.argmax(x1, dim=1)
    y2 = torch.argmax(x2, dim=1)
    return (y1 <= y2).float()


def compute_probabilities(x, apply_softmax=False):
    x1, x2 = x[..., :10], x[..., 10:]
    if apply_softmax:
        x1 = F.softmax(x1)
        x2 = F.softmax(x2)
    return torch.triu(torch.einsum('ij,ik->ijk', x1, x2)).sum()


class Siamese(nn.Module):

    def __init__(self, conv_block_parameters=None, fc_parameters=None, predict=build_predictFC()):
        super().__init__()

        if conv_block_parameters is None:
            conv_block_parameters = {}
        if fc_parameters is None:
            fc_parameters = {}

        self.conv_block_parameters = conv_block_parameters
        self.conv_block = ConvBlock(**conv_block_parameters)

        self.flatten = nn.Flatten()

        self.fc_block_parameters = fc_parameters
        self.fc_block = FCBlock(**fc_parameters)

        self.predict = predict

    def forward_once(self, x):
        y = self.conv_block(x)
        y = self.flatten(y)
        y = self.fc_block(y)
        return y

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        y = torch.cat((y1, y2), dim=1)
        y = self.predict(y)
        return y, (y1, y2)


class CNN(nn.Module):

    def __init__(self, conv_block_parameters, fc_parameters, predict=build_predictFC()):
        super().__init__()
        self.conv_block_parameters = conv_block_parameters
        self.conv_block_parameters['ch_in'] = conv_block_parameters.get(
            'ch_in', 2)
        self.conv_block = ConvBlock(**conv_block_parameters)

        self.fc_block_parameters = fc_parameters
        self.fc_block_parameters['out'] = fc_parameters.get('out', 20)
        self.fc_block = FCBlock(**fc_parameters)

        self.predict = predict

    def forward(self, x):
        y = self.conv_block1(x)
        y = y.flatten()
        y = self.fc_block(y)
        y1, y2 = y[:, 0], y[:, 1]
        y = self.predict(y)
        return y, (y1, y2)


def create_network(input_size, output_size, hidden_layers: list = None,
                   activation=nn.ReLU(),
                   dropout=0.0, use_batch_norm=True):
    if hidden_layers is None:
        hidden_layers = [512]
    hidden_layers = [input_size] + hidden_layers + [output_size]
    layers = []
    for i in range(len(hidden_layers) - 1):
        layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        if i < len(hidden_layers) - 2:
            layers.append(activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class FC(nn.Module):

    def __init__(self, fc_parameters, predict=build_predictFC()):
        super().__init__()
        self.fc_parameters = fc_parameters
        self.fc = create_network(**fc_parameters)
        self.predict = predict

    def forward(self, x):
        y = self.fc(x)
        y1, y2 = y[:, 0], y[:, 1]
        y = self.predict(y)
        return y, (y1, y2)


############################################################################################################
# Raphael

class BaseLineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(392, 150)
        self.fc2 = nn.Linear(150, 80)
        # 10 classes but 2 inputs images => 20 outputs
        self.fc_classes = nn.Linear(80, 20)
        # 1 output: answer to the question: is the first digit less or equal to the second digit?
        self.fc_compare = nn.Linear(20, 1)

    def forward(self, x):
        # flatten view of the inputs
        x = x.view(x.size(0), -1).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # apply a sigmoid (if we want a soft wax we should apply it separately to each half of the output)
        classes = torch.sigmoid(self.fc_classes(x))
        # split the output in 2 parts
        c1 = classes[:, :10]
        c2 = classes[:, 10:]
        # to get the comparison, we need to use the predicted classes
        x = torch.sigmoid(self.fc_compare(classes))
        classes = (c1, c2)
        # we output the prediction and the classes if we want to use auxiliary loss
        return x, classes
