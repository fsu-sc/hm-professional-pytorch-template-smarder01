import torch
import torch.nn as nn
import sys
sys.path.append('/Users/sydneymarder/Desktop/homework #8')
print(sys.path)
from base.base_model import BaseModel

class DenseModel(BaseModel):  # Inherit from BaseModel
    def __init__(self, hidden_layers=1, neurons_per_layer=1, activation_hidden="relu", activation_output="linear"):
        super().__init__()  # Call BaseModel's constructor

        # Define a list to hold layers
        layers = []

        # Input layer
        input_dim = 1
        output_dim = neurons_per_layer if hidden_layers > 0 else 1

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, output_dim))
            if activation_hidden != "linear":
                layers.append(self.get_activation(activation_hidden))
            input_dim = output_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))  # Output layer always has 1 dimension
        if activation_output != "linear":
            layers.append(self.get_activation(activation_output))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def get_activation(self, activation_name):
        # Returns the activation function corresponding to the given name
        activation_map = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "linear": nn.Identity()
        }
        # You can raise an error here to help identify unsupported activations
        if activation_name not in activation_map:
            raise ValueError(f"Activation function '{activation_name}' is not supported.")
        return activation_map[activation_name]
    
    def forward(self, x):
        # Pass input through the defined layers
        return self.model(x)
    