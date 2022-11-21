################ Imports ################
import torch
import numpy
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue
############################################
# ---- Functions --------------------------

# Function to compute the number of parameters in a model
def nb_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    else:
        return sum(parameter.numel() for parameter in model.parameters())

# Function to train a model
def train_model(model, train_input, train_target, train_classes, mini_batch_size, nb_epochs=100, lr=1e-1, criterionTarget=nn.BCEWithLogitsLoss(), criterionClass=nn.CrossEntropyLoss(), w_target=1):
    w_classes = 1 - w_target # weight for the classification loss
    # use an optimizer to update the parameters (gradient descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # iterate over the number of epochs
    for e in range(nb_epochs):
        acc_loss = 0 # accumulate the loss
        # iterate over the number of mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            # get the mini-batch inputs and targets
            mini_batch_input = train_input.narrow(0, b, mini_batch_size)
            mini_batch_target = train_target.narrow(0, b, mini_batch_size)
            mini_batch_classes = train_classes.narrow(0, b, mini_batch_size)
            # compute the output and the loss for the mini-batch
            output, classes_pred = model(mini_batch_input)
            loss_target = criterionTarget(output, mini_batch_target)
            # loss on the classes prediction
            loss_classes = criterionClass(classes_pred, mini_batch_classes)
            # total loss
            loss = w_target * loss_target + w_classes * loss_classes
            # compute the accumulated loss
            acc_loss = acc_loss + loss.item()
            # gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch: {e} loss: {acc_loss}", end="\r", flush=True)

def compute_nb_errors(prediction, target, binary=False):
    if binary:
        # we have only 2 classes and one output value
        # use a threshold to get the predicted class
        pred_classes = (prediction > 0.5)
        return (pred_classes != target).sum().item()
    else:
        # use a "winner-takes-all" rule to compute the number of prediction errors
        _, predicted_classes = prediction.max(1) # get the index of the max for each row
        _, target_classes = target.max(1)
        nb_errors = (predicted_classes != target_classes).sum().item()
        return nb_errors

############################################
# ---- Define the models ------------------

# Simple model with fully connected layers
class BaseLineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(392, 150)
        self.fc2 = nn.Linear(150, 80)
        self.fc_classes = nn.Linear(80, 20) # 10 classes but 2 inputs images => 20 outputs
        self.fc_compare = nn.Linear(20, 1) # 1 output: answer to the question: is the first digit less or equal to the second digit?


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # apply a sigmoid (if we want a soft wax we should apply it separately to each half of the output)
        classes = torch.sigmoid(self.fc_classes(x))
        # to get the comparison, we need to use the predicted classes
        x = torch.sigmoid(self.fc_compare(classes))
        # we output the prediction and the classes if we want to use auxiliary loss
        return x, classes

# siamese convolutional network
class SiameseConvNet1(nn.Module):
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

        # fully connected layers to get the classes prediction and the target prediction
        self.fcNet = nn.Sequential(
            nn.Linear(20, 20),
            nn.Linear(20, 1), # 2 inputs images and 10 classes => 2x10 
            # 1 output: answer to the question: is the first digit less or equal to the second digit?
            nn.Sigmoid() # sigmoid to get an answer between 0 and 1
        )


    def forward(self, x):
        # apply the convolutional layers
        c1 = self.convNet(x[:, 0, :, :].unsqueeze(1)) # unsqueeze to add the channel dimension: N x 1 x 14 x 14
        c2 = self.convNet(x[:, 1, :, :].unsqueeze(1))
        # size of c1 and c2: N x 10
        # concatenate the classes prediction
        classes = torch.cat((c1.unsqueeze(1), c2.unsqueeze(1)), dim=1) # N x 2 x 10
        # apply the fully connected layers to get the target prediction
        x = self.fcNet(classes.view(-1, 20))
        # we output the prediction and the classes if we want to use auxiliary loss
        return x, classes

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
        return x, classes
############################################
# ---- Test -------------------------------

if __name__ == "__main__":
    # number of pairs
    N_pairs = 1000
    # generate the pairs
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N_pairs)
    train_target = train_target.view(-1, 1)
    test_target = test_target.view(-1, 1)
    # classes to one-hot encoding
    train_classes = F.one_hot(train_classes)
    test_classes = F.one_hot(test_classes)
    # print the shapes
    print(f"train_input shape: {train_input.shape}, type: {train_input.dtype}")
    print(f"train_target shape: {train_target.shape}, type: {train_target.dtype}")
    print(f"train_classes shape: {train_classes.shape}, type: {train_classes.dtype}")
    # flatten view of the inputs
    train_input_flat = train_input.view(train_input.size(0), -1).float()
    test_input_flat = test_input.view(test_input.size(0), -1).float()
    # flatten view of the classes = convert to one-hot encoding + concat class1 and class2
    train_classes_flat = train_classes.view(train_classes.size(0), -1).float()
    test_classes_flat = test_classes.view(test_classes.size(0), -1).float()
    print(f"train_input_flat shape: {train_input_flat.shape}, type: {train_input_flat.dtype}")
    print(f"train_classes_flat shape: {train_classes_flat.shape}, type: {train_classes_flat.dtype}")

    ####### Baseline model
    print("\n##### Baseline model #####")
    print(f"number of parameters for the baseline model: {nb_parameters(BaseLineNet())}")
    # train the baseline model
    baseline = BaseLineNet()
    train_model(baseline, train_input_flat, train_target.float(), train_classes_flat, mini_batch_size=25, nb_epochs=25, lr=1e-1, criterionTarget=nn.BCELoss(), criterionClass=nn.CrossEntropyLoss(), w_target=0.8)

    # test the baseline model
    output, classes_pred = baseline(test_input_flat)
    nb_errors = compute_nb_errors(output, test_target, binary=True)
    print(f"\nnumber of errors: {nb_errors} / {test_input_flat.size(0)}, error rate: {nb_errors/test_input_flat.size(0)*100:.2f}%")

    # siamese model 1
    print("\n##### Siamese model 1 #####")
    siameseConv = SiameseConvNet1()
    print(f"number of parameters for the siamese model 1: {nb_parameters(siameseConv)}")
    
    # train the siamese model
    train_model(siameseConv, train_input, train_target.float(), train_classes.float(), mini_batch_size=25, nb_epochs=25, lr=3e-2, criterionTarget=nn.BCELoss(), criterionClass=nn.CrossEntropyLoss(), w_target=0.7)

    # test the siamese model
    output, classes_pred = siameseConv(test_input)
    nb_errors = compute_nb_errors(output, test_target, binary=True)
    print(f"\nnumber of errors: {nb_errors} / {test_input.size(0)}, error rate: {nb_errors/test_input.size(0)*100:.2f}%")

    # siamese model 2
    print("\n##### Siamese model 2 #####")
    siameseConv = SiameseConvNet2()
    print(f"number of parameters for the siamese model 2: {nb_parameters(siameseConv)}")
    
    # train the siamese model
    train_model(siameseConv, train_input, train_target.float(), train_classes.float(), mini_batch_size=25, nb_epochs=25, lr=1e-1, criterionTarget=nn.BCEWithLogitsLoss(), criterionClass=nn.CrossEntropyLoss(), w_target=0.1)

    # test the siamese model
    output, classes_pred = siameseConv(test_input)
    nb_errors = compute_nb_errors(output, test_target, binary=True)
    print(f"\nnumber of errors: {nb_errors} / {test_input.size(0)}, error rate: {nb_errors/test_input.size(0)*100:.2f}%")

    

