import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, config["h1"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["h1"], config["h2"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["h2"], 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class cnn(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.in_channels = config["input_size"][1]
        self.input_size = config["input_size"]

        self.convolutions = nn.Sequential(
            nn.Conv2d(
                self.in_channels, config["filters"], kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                config["filters"], config["filters"], kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                config["filters"], config["filters"], kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        activation_map_size = self._conv_test(self.input_size)

        self.agg = nn.AvgPool2d(activation_map_size)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config["filters"], config["h1"]),
            nn.ReLU(),
            nn.Linear(config["h1"], config["h2"]),
            nn.ReLU(),
            nn.Linear(config["h2"], 10),
        )

    def _conv_test(self, input_size):
        x = torch.ones(input_size, dtype=torch.float32)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits
