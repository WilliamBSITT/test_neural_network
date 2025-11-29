from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from nn.activation import Activation
from nn.callback import Callback
from nn.layer import Layer
from nn.loss import Loss
from nn.optimizer import Optimizer


class Model(ABC):
    """Abstract base model class"""

    @property
    @abstractmethod
    def learning_rate(self): ...

    @learning_rate.setter
    @abstractmethod
    def learning_rate(self, value: float): ...

    @abstractmethod
    def __call__(self, input_tensor: NDArray) -> NDArray: ...

    @abstractmethod
    def fit(
        self,
        examples: NDArray,
        labels: NDArray,
        epochs: int,
        verbose: bool = False,
        callbacks: Tuple[Callback] = (),
        log_interval: int = 1,
    ): ...

    @abstractmethod
    def predict(self, examples: NDArray) -> NDArray: ...

    @abstractmethod
    def evaluate(self, examples: NDArray, labels: NDArray) -> NDArray: ...

    @abstractmethod
    def backward_step(self, labels: NDArray): ...

    @abstractmethod
    def update(self): ...


class NeuralNetwork(Model):
    def __init__(
        self,
        layers: Tuple[Tuple[Layer, Activation]],
        loss: Loss,
        optimizer: Optimizer,
        regularization_factor: float = 0.0,
    ):
        self._layers = layers
        self._num_layers = len(layers)
        self._loss = loss
        self._optimizer = optimizer
        self._regularization_factor = regularization_factor
        self._input = None
        self._output = None
        self._num_examples = None

    def __call__(self, input_tensor: NDArray) -> NDArray:
        if self._num_examples is None:
            self._num_examples = input_tensor.shape[-1]

        output = input_tensor

        for layer, activation in self._layers:
            output = layer(output)
            output = activation(output)

        self._output = output
        return self._output

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value

    def backward_step(self, labels: NDArray):
        da = self._loss.gradient(self._output, labels)

        for index in reversed(range(0, self._num_layers)):
            layer, activation = self._layers[index]

            if index == 0:
                prev_layer_output = self._input
            else:
                prev_layer, prev_activation = self._layers[index - 1]
                prev_layer_output = prev_activation(prev_layer.output)

            dz = np.multiply(da, activation.gradient(layer.output))
            layer.grad_weights = (
                np.dot(dz, np.transpose(prev_layer_output)) / self._num_examples
            )
            layer.grad_weights = (
                layer.grad_weights
                + (self._regularization_factor / self._num_examples) * layer.weights
            )
            layer.grad_bias = np.mean(dz, axis=1, keepdims=True)
            da = np.dot(np.transpose(layer.weights), dz)

    def fit(
        self,
        examples: NDArray,
        labels: NDArray,
        epochs: int,
        verbose: bool = False,
        callbacks: Tuple[Callback] = (),
        validation_data: tuple[NDArray, NDArray] | None = None,
        log_interval: int = 1,
        initial_epoch: int = 0,
    ):
        # attach model to callbacks if they need it
        for callback in callbacks:
            try:
                callback.model = self
            except Exception:
                pass

        start = initial_epoch + 1
        end = initial_epoch + epochs

        for epoch in range(start, end + 1):
            self._input = examples
            _ = self(self._input)
            loss = self._loss(self._output, labels)
            self.backward_step(labels)
            self.update()

            # compute validation loss if requested
            val_loss = None
            if validation_data is not None:
                val_x, val_y = validation_data
                val_loss = float(np.squeeze(self.evaluate(val_x, val_y)))

            # notify callbacks with validation loss when available, else training loss
            loss_scalar = float(np.squeeze(val_loss if val_loss is not None else loss))
            for callback in callbacks:
                callback.on_epoch_end(epoch, loss_scalar)

            if verbose and epoch % log_interval == 0:
                print(f"Epoch: {epoch:05d} / {epochs}, Loss {loss_scalar:0.5f}")

            # check for stop signal from callbacks
            stop = any(getattr(cb, 'stop_training', False) for cb in callbacks)
            if stop:
                break

    def predict(self, examples: NDArray) -> NDArray:
        outputs = self(examples)
        return outputs

    def evaluate(self, examples: NDArray, labels: NDArray) -> NDArray:
        _ = self(examples)
        return self._loss(self._output, labels)

    def update(self):
        for ln in range(0, len(self._layers)):
            self._optimizer._layer_number = ln
            self._layers[ln][0].update(self._optimizer)

    def save(self, file_path: str):
        import pickle

        # Temporarily convert any arrays to NumPy for safe pickling
        original_weights = []
        for layer, _ in self._layers:
            original_weights.append((layer.weights, layer.bias))
            if layer.weights is not None:
                layer.weights = np.asarray(layer.weights)
            if layer.bias is not None:
                layer.bias = np.asarray(layer.bias)

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
        finally:
            # Restore original arrays (possibly on GPU)
            for (layer, _), (w, b) in zip(self._layers, original_weights):
                layer.weights = w
                layer.bias = b

    @staticmethod
    def load(file_path: str):
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
