import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, config: dict):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(config["input_size"], config["h1"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["h1"], config["h2"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config["h2"], config["output_size"]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Convolutional block with optional dropout and maxpool."""
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        ]

        layers.append(nn.MaxPool2d(kernel_size=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        """Linear block with ReLU activation and optional dropout."""
        super(LinearBlock, self).__init__()
        layers = [
            nn.Linear(in_features, out_features),
            nn.ReLU()
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



class cnn(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.input_size = config["input_size"]

        self.convolutions = nn.ModuleList()

        #TODO ombouwen filterbak
        for i, filters in enumerate(config["conv_blocks"]):
            in_ch = self.input_size[1] if i == 0 else config["conv_blocks"][i-1]
            self.convolutions.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=filters,
                )
            )
        self.dropout = nn.Dropout(config["dropout"])
        
        activation_map_size = self._conv_test(self.input_size)

        self.agg = nn.AvgPool2d(activation_map_size)

        self.dense_layers = nn.ModuleList()

        self.flatten = nn.Flatten()
        # Loop to create linear blocks based on config["linear_blocks"]
        for i, blockconfig in enumerate(config["linear_blocks"]):
            in_features = config["conv_blocks"][-1] if i == 0 else config["linear_blocks"][i-1]["out_features"]
            self.dense_layers.append(
                LinearBlock(
                    in_features=in_features,
                    out_features=blockconfig["out_features"],
                    dropout=blockconfig.get("dropout", 0.0)
                )
            )

        # Final output layer (no activation)
        self.output_layer = nn.Linear(config["linear_blocks"][-1]["out_features"], config["output_size"])

    def _conv_test(self, input_size):
        x = torch.ones(input_size, dtype=torch.float32)
        for conv in self.convolutions:  # Apply each block in the ModuleList
            x = conv(x)
        return x.shape[-2:]

    def forward(self, x):
        for conv in self.convolutions:
            x = conv(x)
        x = self.dropout(x)
        x = self.agg(x)
        x = self.flatten(x)
        for dense in self.dense_layers:
            x = dense(x)     
        logits = self.output_layer(x)
        return logits
