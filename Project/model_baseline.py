################ Imports ################
import torch
import numpy
from torch import Tensor
from torch import nn
from torch.nn import functional as F
############################################

# Simple model with fully connected layers
class BaseLineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(392, 150)
        self.fc2 = nn.Linear(150, 80)
        self.fc_classes = nn.Linear(80, 20) # 10 classes but 2 inputs images => 20 outputs
        self.fc_compare = nn.Linear(20, 1) # 1 output: answer to the question: is the first digit less or equal to the second digit?


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
        return x.flatten(), classes
