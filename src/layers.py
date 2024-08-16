import numpy as np

EPSILON = 1e-12
"""Linear Layer"""


# Linear Layer <w, X> + b
class Linear:
    def __init__(self, num_inputs, num_outputs, bias_initializer=None):
        bound = 1 / np.sqrt(num_inputs) if num_inputs > 1 else 0
        self.weights = np.random.uniform(
            low=-bound, high=bound, size=(num_inputs, num_outputs)
        )
        if bias_initializer is None:
            self.biases = np.random.uniform(
                low=-bound, high=bound, size=((1, num_outputs))
            )
        else:
            self.biases = np.ones(shape=(1, num_outputs)) * bias_initializer

    def forward(self, inputs):
        self.inputs = inputs
        self.activations = np.dot(inputs, self.weights) + self.biases
        return self.activations

    def backward(self, delta_inputs):
        self.wgradients = np.dot(self.inputs.T, delta_inputs)
        self.bgradients = np.sum(delta_inputs, axis=0, keepdims=True)
        self.grads_inputs = np.dot(delta_inputs, self.weights.T)


# Linear Layer <w, X> + b
class Linear:
    def __init__(
        self, num_inputs, num_outputs, weights_init=None, bias_initializer=None
    ):
        bound = 1 / np.sqrt(num_inputs) if num_inputs > 1 else 0
        if weights_init is None:
            self.weights = np.random.uniform(
                low=-bound, high=bound, size=(num_inputs, num_outputs)
            )
        else:
            self.weights = weights_init
        if bias_initializer is None:
            self.biases = np.random.uniform(
                low=-bound, high=bound, size=((1, num_outputs))
            )
        else:
            self.biases = bias_initializer

    def forward(self, inputs):
        self.inputs = inputs
        self.activations = np.dot(inputs, self.weights) + self.biases

    def backward(self, delta_inputs):
        self.wgradients = np.dot(self.inputs.T, delta_inputs)
        self.bgradients = np.sum(delta_inputs, axis=0, keepdims=True)
        self.grads_inputs = np.dot(delta_inputs, self.weights.T)


"""Activation Funtions"""


# ReLU activation
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.activations = np.maximum(0, inputs)
        return self.activations

    def backward(self, delta_inputs):
        self.grads_inputs = delta_inputs.copy()
        self.grads_inputs[self.inputs <= 0] = 0


# ReLU activation
class Square:
    def forward(self, inputs):
        self.inputs = inputs
        self.activations = inputs**2
        return self.activations

    def backward(self, delta_inputs):
        self.grads_inputs = (2 * self.inputs) * delta_inputs


# Sigmoid activation
class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.activations = self.__operation(self.inputs)
        return self.activations

    def __operation(self, inputs):
        exps = 1 + np.exp(-inputs)
        return 1 / exps

    def backward(self, delta_inputs):
        grads = self.__operation(self.inputs) * (1 - self.__operation(self.inputs))
        self.grads_inputs = grads * delta_inputs


"""Loss Functions"""


class BinaryCrossEntropy:
    def __init__(self, class_weights=None):
        self.weights = class_weights
        if self.weights is None:
            self.weights = {0: 1, 1: 1}

    def forward(self, y_true, y_pred):
        loss_value = self.weights[0] * (y_true - 1) * np.clip(
            np.log(1 - y_pred), a_min=-100, a_max=None
        ) - self.weights[1] * y_true * np.clip(np.log(y_pred), a_min=-100, a_max=None)
        return np.mean(loss_value)

    def backward(self, y_true, y_pred):
        num_samples = len(y_true)
        self.grads_inputs = (
            self.weights[0] * y_pred
            - self.weights[1] * y_true
            + (self.weights[1] - self.weights[0]) * y_pred * y_true
        ) / np.clip((1 - y_pred) * y_pred, a_min=EPSILON, a_max=None)
        self.grads_inputs /= num_samples


# mean squared error loss
class MeanSquaredError:
    def forward(self, y_true, y_pred):
        losses = 0.5 * (y_true - y_pred) ** 2
        self.loss = np.mean(losses)
        return self.loss

    def backward(self, y_true, y_pred):
        num_samples = len(y_pred)
        self.grads_inputs = y_true - y_pred
        self.grads_inputs = self.grads_inputs / num_samples
