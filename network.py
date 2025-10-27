import torch
import torch.nn as nn

# fully-connected feedforward neural network
class net(nn.Module):
    def __init__(self, n_in, n_l, n_n, n_out, activation):
        super(net,self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(n_in, n_n))
        self.layers.append(activation())
        # Hidden layers
        for _ in range(n_l):
            self.layers.append(nn.Linear(n_n, n_n))
            self.layers.append(activation())
        # Output layer
        self.layers.append(nn.Linear(n_n, n_out))

        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)  
        
    # forward passing through network
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x