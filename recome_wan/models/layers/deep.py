import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Union


def set_activation(activation: str or nn.Module) -> nn.Module:
    """
    Create a new instance of PyTorch activation function.

    Args:
    activation (str or nn.Module): A string representing the name of the activation function or a PyTorch activation
    function itself.

    Returns:
    A PyTorch activation function object.

    Raises:
    TypeError: If `activation` is not a string or a PyTorch activation function object.
    ValueError: If `activation` is a string that does not represent a valid PyTorch activation function name.
    """

    if isinstance(activation, str):
        # If `activation` is a string, convert it to lowercase and check whether it represents a valid PyTorch
        # activation function name.
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            # If `activation` is a string that does not represent a valid PyTorch activation function name,
            # raise a ValueError.
            raise ValueError(f"{activation} is not a valid PyTorch activation function name.")
    elif isinstance(activation, nn.Module):
        # If `activation` is already a PyTorch activation function object, return it as is.
        return activation
    else:
        # If `activation` is not a string or a PyTorch activation function object, raise a TypeError.
        raise TypeError("`activation` must be a string or a PyTorch activation function object.")

#DNN


class MLP_Layer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: Optional[int] = None,
            hidden_units: List[int] = [],
            hidden_activations: Union[str, List[str]] = "ReLU",
            final_activation: Optional[str] = None,
            dropout_rates: Union[float, List[float]] = 0,
            batch_norm: bool = False,
            use_bias: bool = True,
    ):
        """
        Initialization of a Multi-Layer Perceptron.

        Args:
            input_dim: An integer representing the dimension of the input of the first layer.
            output_dim: An integer representing the dimension of the output of the final layer.
            hidden_units: A list of integers representing the dimensions of the hidden layers.
            hidden_activations: A string or a list of strings representing the activation function(s) of the hidden layers.
            final_activation: A string representing the activation function of the final layer.
            dropout_rates: A float or a list of floats representing the dropout rate(s) of the hidden layers.
            batch_norm: A boolean indicating whether to use Batch Normalization or not.
            use_bias: A boolean indicating whether to use bias or not.
        """
        super(MLP_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [set_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if final_activation is not None:
            dense_layers.append(set_activation(final_activation))
        self.dnn = nn.Sequential(*dense_layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Compute the forward pass of the Multi-Layer Perceptron.

        Args:
            inputs: A tensor. It must have the same dimensionality as the input dimension specified during initialization.

        Returns:
            A tensor. It has the same dimensionality as the output dimension specified during initialization, if there is any.
        """
        return self.dnn(inputs)