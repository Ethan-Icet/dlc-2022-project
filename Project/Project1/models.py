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


########################################################################################################################
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


# siamese convolutional network
class SiameseConvNet1(nn.Module):
    def __init__(self):
        super().__init__()
        # first we define the shared cnn part
        self.convNet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),  # image size: 14x14 => 12x12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # image size: 12x12 => 6x6
            nn.Conv2d(64, 80, kernel_size=3),  # image size: 6x6 => 4x4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # image size: 4x4 => 2x2
            nn.Conv2d(80, 64, kernel_size=2),  # image size: 2x2 => 1x1
            nn.Flatten(),  # flatten the output: N x 64 x 1 x 1 => N x 64
            nn.ReLU(),
            # linear layer to get the classes
            nn.Linear(64, 32),
            nn.Linear(32, 10),  # 10 classes
            nn.Softmax(dim=1),  # softmax to get a probability distribution over the classes
        )

        # fully connected layers to get the classes prediction and the target prediction
        self.fcNet = nn.Sequential(
            nn.Linear(20, 20),
            nn.Linear(20, 1),  # 2 inputs images and 10 classes => 2x10
            # 1 output: answer to the question: is the first digit less or equal to the second digit?
            nn.Sigmoid()  # sigmoid to get an answer between 0 and 1
        )

    def forward(self, x):
        # apply the convolutional layers
        c1 = self.convNet(x[:, 0, :, :].unsqueeze(1))  # unsqueeze to add the channel dimension: N x 1 x 14 x 14
        c2 = self.convNet(x[:, 1, :, :].unsqueeze(1))
        # size of c1 and c2: N x 10
        # concatenate the classes prediction
        classes = torch.cat((c1.unsqueeze(1), c2.unsqueeze(1)), dim=1)  # N x 2 x 10
        # apply the fully connected layers to get the target prediction
        x = self.fcNet(classes.view(-1, 20))
        # we output the prediction and the classes if we want to use auxiliary loss
        return x, (c1, c2)


# siamese convolutional network with prior knowledge on comparison
class SiameseConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        # first we define the shared cnn part
        self.convNet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),  # image size: 14x14 => 12x12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # image size: 12x12 => 6x6
            nn.Conv2d(64, 80, kernel_size=3),  # image size: 6x6 => 4x4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # image size: 4x4 => 2x2
            nn.Conv2d(80, 64, kernel_size=2),  # image size: 2x2 => 1x1
            nn.Flatten(),  # flatten the output: N x 64 x 1 x 1 => N x 64
            nn.ReLU(),
            # linear layer to get the classes
            nn.Linear(64, 32),
            nn.Linear(32, 10),  # 10 classes
            nn.Softmax(dim=1),  # softmax to get a probability distribution over the classes
        )

    def forward(self, x):
        # apply the convolutional layers
        c1 = self.convNet(x[:, 0, :, :].unsqueeze(1))  # unsqueeze to add the channel dimension: N x 1 x 14 x 14
        c2 = self.convNet(x[:, 1, :, :].unsqueeze(1))
        # size of c1 and c2: N x 10
        # concatenate the classes prediction
        classes = torch.cat((c1.unsqueeze(1), c2.unsqueeze(1)), dim=1)  # N x 2 x 10
        # computes the matrix of joint probabilities
        joint_prob = torch.einsum('ij,ik->ijk', c1, c2)  # N x 10 x 10
        # takes the upper triangular part of the matrix
        upper_tri = torch.triu(joint_prob)  # N x 10 x 10
        # sums the probabilities
        x = torch.sum(upper_tri, dim=(1, 2)).unsqueeze(1)
        # N x 1 => probability that the first digit is less or equal to the second digit
        return torch.clamp(x, 0, 1), (c1, c2)


########################################################################################################################
# Kathleen

class SiameseNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # 1, 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 16, 8
        self.maxpool_2d = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5 * 5 * 32, 160)
        self.fc2 = nn.Linear(160, 10)

        self.lin1 = nn.Linear(2, 20)
        self.lin2 = nn.Linear(20, 2)

    def forward_once(self, x1):
        x = F.relu(self.conv1(x1))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.dropout(x, p=0.5)#, training=training)

        x = x.view(-1, 5 * 5 * 32)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Changed from log_softmax because negative otherwise

        return x

    def forward(self, x):
        x1 = x[:, 0].view(x.size(0), 1, 14, 14)
        x2 = x[:, 1].view(x.size(0), 1, 14, 14)

        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)  # torch.Size([50, 10])

        z1 = torch.argmax(y1, 1)
        z2 = torch.argmax(y2, 1)

        x = torch.cat((z1.view(-1, z1.size(0)), z2.view(-1, z2.size(0))), axis=0).t().float()

        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        x = F.softmax(x, dim=1)  # Changed from log_softmax because negative otherwise

        return x, (y1, y2)


class SiameseNNAll(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # 1, 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 16, 8
        self.maxpool_2d = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5 * 5 * 32, 256)
        self.fc2 = nn.Linear(256, 10)

        self.lin1 = nn.Linear(2, 20)
        self.lin2 = nn.Linear(20, 2)

    def forward_once(self, x1):
        x = F.relu(self.conv1(x1))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.dropout(x, p=0.5)#, training=training)

        x = x.view(-1, 5 * 5 * 32)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Changed from log_softmax because negative otherwise

        return x

    def forward(self, x):
        x1 = x[:, 0].view(x.size(0), 1, 14, 14)
        x2 = x[:, 1].view(x.size(0), 1, 14, 14)

        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)  # torch.Size([50, 10])

        z1 = torch.argmax(y1, 1)
        z2 = torch.argmax(y2, 1)

        x = torch.cat((z1.view(-1, z1.size(0)), z2.view(-1, z2.size(0))), axis=0).t().float()

        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        x = F.softmax(x, dim=1)  # Changed from log_softmax because negative otherwise

        return x, (y1, y2)


class Siamese2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # (1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # (32, 64, kernel_size=3)
        self.maxpool_2d = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)  # (5*5*64, 256)
        self.fc2 = nn.Linear(120, 10)  # (256, 10)

    def forward_once(self, x1):
        x = F.relu(self.conv1(x1))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.dropout(x, p=0.5)#, training=training)

        x = x.view(-1, 5 * 5 * 16)  # (-1, 5*5*64 )

        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Changed from log_softmax because negative otherwise
        return x

    def forward(self, x):
        x1 = x[:, 0].view(x.size(0), 1, 14, 14)
        x2 = x[:, 1].view(x.size(0), 1, 14, 14)
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)  # torch.Size([50, 10])
        return None, (y1, y2)


class SiameseAll(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.maxpool_2d = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5 * 5 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward_once(self, x1):
        x = F.relu(self.conv1(x1))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.dropout(x, p=0.5)#, training=training)

        x = x.view(-1, 5 * 5 * 64)

        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Changed from log_softmax because negative otherwise
        return x

    def forward(self, x):
        x1 = x[:, 0].view(x.size(0), 1, 14, 14)
        x2 = x[:, 1].view(x.size(0), 1, 14, 14)

        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)  # torch.Size([50, 10])
        return None, (y1, y2)
