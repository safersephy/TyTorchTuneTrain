from torch import Tensor, nn
from loguru import logger

class RNN(nn.Module):
    def __init__(self, config: dict) -> None:
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
    def __init__(self, config: dict) -> None:
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
    
    
class LSTMConv(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()     
        self.conv = nn.Conv1d(in_channels=config["input_size"], out_channels=12, kernel_size=3) 
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(12)    
        self.dropout = nn.Dropout(.3)   
        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=config["hidden_size"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(config["hidden_size"], config["horizon"])
        self.horizon = config["horizon"]

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x.transpose(1, 2))
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        
        x, (h_n, c_n) = self.lstm(x.transpose(1, 2))
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat
