class Module(object):
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grads_wrt_output):
        raise NotImplementedError

    def param(self) -> list:
        return []
