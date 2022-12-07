import torch
from module import *
from linear import  Linear

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
  

    def forward(self, x):
        # save the input to use it later in the backward pass
        self.x = x.clone() # the input is of shape (batch_size, ..., ..., etc)
        # The forward pass is very simple. We just call the forward method of each module one after the other
        for module in self.modules:
            x = module.forward(x)
        return x # return the output of the last module

    def backward(self, grad):
        # the backward takes the gradient of the loss with respect to the output of the last module.
        # The last module is called first with the gradient of the loss with respect to its output and we obtain the
        # gradient of the loss with respect to its input. We then call the backward method of the second last module
        # with the gradient of the loss with respect to its input and so on (We call the backward method of each module
        # in reverse order with the gradient obtained from the previous module).
        for module in self.modules[::-1]: # we call the backward method of each module in reverse order
            grad = module.backward(grad)
        return grad # return the gradient of the loss with respect to the input of the first module
       

    def param(self) -> list:
        # return a list of all parameters of the model
        return [param for module in self.modules for param in module.param()]

    def __repr__(self):
        s = "Sequential(\n"
        for module in self.modules:
            s += str(module)
            s += ",\n"
        s += ")"
        return s


# --- test ---
if __name__ == "__main__":
    input_size = 3
    output_size = 3
    nb_hidden = 4
    linear_custom_list = [Linear(input_size, output_size) for _ in range(nb_hidden)]
    linear_torch_list = [torch.nn.Linear(input_size, output_size) for _ in range(nb_hidden)]
    # make sure the parameters are the same
    with torch.no_grad():
        for i in range(nb_hidden):
            linear_torch_list[i].weight = torch.nn.Parameter(linear_custom_list[i].w.t())
            linear_torch_list[i].bias = torch.nn.Parameter(linear_custom_list[i].b)
    # create the model
    seq_custom = Sequential(*linear_custom_list)
    seq_torch = torch.nn.Sequential(*linear_torch_list)
    # test forward
    batch_size = 6
    x = torch.randn(batch_size, input_size, requires_grad=True) # random input
    y_custom = seq_custom.forward(x)
    y_torch = seq_torch.forward(x)
    # make sure the outputs are the same
    print(f"Are the outputs the same? {torch.allclose(y_custom, y_torch)}")

    # backward
    grad = torch.randn(batch_size, output_size) # random gradient
    grad_custom = seq_custom.backward(grad)
    y_torch.backward(grad)
    grad_torch = x.grad
    # print(grad_torch, grad_custom)
    # make sure the gradients are the same
    print(f"Are the gradients the same? {torch.allclose(grad_custom, grad_torch)}")