import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=4096):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.encoder(x))
        x_hat = self.decoder(h)
        return h, x_hat

    def encode(self, x):
        return self.relu(self.encoder(x))

    def decode(self, h):
        return self.decoder(h)
