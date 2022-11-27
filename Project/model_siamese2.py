################ Imports ################
import torch
import numpy
from torch import Tensor
from torch import nn
from torch.nn import functional as F
############################################

# siamese convolutional network with prior knowledge on comparison
class SiameseConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        # first we define the shared cnn part
        self.convNet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3), # image size: 14x14 => 12x12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # image size: 12x12 => 6x6
            nn.Conv2d(64, 80, kernel_size=3), # image size: 6x6 => 4x4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # image size: 4x4 => 2x2
            nn.Conv2d(80, 64, kernel_size=2), # image size: 2x2 => 1x1
            nn.Flatten(), # flatten the output: N x 64 x 1 x 1 => N x 64
            nn.ReLU(),
            # linear layer to get the classes
            nn.Linear(64, 32),
            nn.Linear(32, 10), # 10 classes
            nn.Softmax(dim=1), # softmax to get a probability distribution over the classes
        )


    def forward(self, x):
        # apply the convolutional layers
        c1 = self.convNet(x[:, 0, :, :].unsqueeze(1)) # unsqueeze to add the channel dimension: N x 1 x 14 x 14
        c2 = self.convNet(x[:, 1, :, :].unsqueeze(1))
        # size of c1 and c2: N x 10
        # concatenate the classes prediction
        classes = torch.cat((c1.unsqueeze(1), c2.unsqueeze(1)), dim=1) # N x 2 x 10
        # computes the matrix of joint probabilities
        joint_prob = torch.einsum('ij,ik->ijk', c1, c2) # N x 10 x 10
        # takes the upper triangular part of the matrix
        upper_tri = torch.triu(joint_prob) # N x 10 x 10
        # sums the probabilities
        x = torch.sum(upper_tri, dim=(1, 2)).unsqueeze(1) # N x 1 => probability that the first digit is less or equal to the second digit
        return x.flatten(), (c1, c2)
