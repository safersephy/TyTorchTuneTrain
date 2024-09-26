import torch
from torch import nn
Tensor = torch.Tensor

class RNN(nn.Module):
    def __init__(
        self, config: dict
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(config["hidden_size"], config["horizon"])
        self.horizon = config["horizon"]

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat

class LSTM(nn.Module):
    def __init__(
        self, config: dict
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(config["hidden_size"], config["horizon"])
        self.horizon = config["horizon"]

    def forward(self, x: Tensor) -> Tensor:
        x, (h_n, c_n) = self.lstm(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat