from module import *
import torch

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # number of input features (dimension of input) and number of output features (dimension of output)
        self.in_features = in_features
        self.out_features = out_features

        # --- weight matrix and bias vector ---
        # weight matrix of size (out_features, in_features) initialized with random values from a normal distribution N(0, 1)
        self.w = torch.randn(in_features, out_features, requires_grad=False)
        # bias vector of size (out_features) initialized at 0
        self.b = torch.zeros(out_features, requires_grad=False)

        # --- gradient of weight matrix and bias vector ---
        # these are the gradients of the loss according to the weights and the bias
        self.w_grad = torch.zeros(in_features, out_features, requires_grad=False)
        self.b_grad = torch.zeros(out_features, requires_grad=False)

        # --- local variables ---
        # we need to store the input x to use it in the backward pass
        # x is a tensor of size (batch_size, in_features)
        self.x = None

    def forward(self, x):
        # first we store the input x
        self.x = x.clone()
        # then we compute the output of the linear layer : y = x * w + b
        # x is a tensor of size (batch_size, in_features)
        # the output y is a tensor of size (batch_size, out_features)
        return self.x @ self.w + self.b

    def backward(self, grad):
        # grad is the gradient of the loss according to the output of the linear layer.
        # grad is a tensor of size (batch_size, out_features)
        # we will:
        # 1) accumulate the gradient of the loss according to the weights and the bias
        # 2) compute the gradient of the loss according to the input of the linear layer
        
        # --- 1) accumulate the gradient of the loss according to the weights and the bias ---
        # x: (batch_size, in_features), grad: (batch_size, out_features)
        # to get the gradient of the loss according to the weights, simply do x.T * grad
        # (in_features, batch_size) @ (batch_size, out_features) = (in_features, out_features)
        self.w_grad += self.x.t() @ grad
        # for the bias, we just need to sum the gradients of the loss according to the bias over the number of batch
        self.b_grad += grad.sum(dim=0)

        # --- 2) compute the gradient of the loss according to the input of the linear layer ---
        # grad: (batch_size, out_features), w: (in_features, out_features)
        # (batch_size, out_features) @ (out_features, in_features) = (batch_size, in_features)
        return grad @ self.w.t() # return a tensor of size (batch_size, in_features)

    def param(self) -> list:
        # return pairs of parameters and gradients
        return [(self.w, self.w_grad), (self.b, self.b_grad)]

    def __repr__(self):
        # fancy print
        return f"Linear({self.in_features}, {self.out_features})"

# --- test ---
if __name__ == "__main__":
    input_size = 3
    output_size = 2
    linear_custom = Linear(input_size, output_size)
    linear_torch = torch.nn.Linear(input_size, output_size)
    # make sure the parameters are the same
    with torch.no_grad():
        linear_torch.weight = torch.nn.Parameter(linear_custom.w.t())
        linear_torch.bias = torch.nn.Parameter(linear_custom.b)
    # forward
    batch_size = 6
    x = torch.randn(batch_size, input_size, requires_grad=True) # random input
    y_custom = linear_custom.forward(x)
    y_torch = linear_torch.forward(x)
    # make sure the outputs are the same
    print(f"Are the outputs the same? {torch.allclose(y_custom, y_torch)}")

    # backward
    grad = torch.randn(batch_size, output_size) # random gradient
    grad_custom = linear_custom.backward(grad)
    y_torch.backward(grad)
    grad_torch = x.grad
    # print(grad_torch, grad_custom)
    # make sure the gradients are the same
    print(f"Are the gradients the same? {torch.allclose(grad_custom, grad_torch)}")
