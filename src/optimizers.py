# stocastic gradient descent
class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def update_params(self, layers):
        for layer in layers:
            if hasattr(layer, "weights"):
                layer.weights -= self.lr * layer.wgradients
            if hasattr(layer, "biases"):
                layer.biases -= self.lr * layer.bgradients
