import warnings

import mlflow
import mlflow.pytorch
from mads_datasets import DatasetFactoryProvider, DatasetType
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torchinfo import summary
from tytorch.trainer import EarlyStopping, Trainer

from models.time_series import LSTM, LSTMConv
from utils.data import add_batch_padding
from utils.mlflow import set_mlflow_experiment

warnings.simplefilter("ignore", UserWarning)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
set_mlflow_experiment("train_ts")

params = {
    "model_class": LSTMConv,
    "batch_size": 32,
    "n_epochs": 50,
    "input_size": 3,
    "hidden_size": 80,
    "num_layers": 3,
    "horizon": 20,
    "lr": 1e-3,
    "dataset_type": DatasetType.GESTURES,
}

datasets = DatasetFactoryProvider.create_factory(
    params["dataset_type"]
).create_dataset()
trainloader = DataLoader(
    datasets["train"],
    batch_size=params["batch_size"],
    shuffle=True,
    collate_fn=add_batch_padding,
)
testloader = DataLoader(
    datasets["valid"],
    batch_size=params["batch_size"],
    shuffle=True,
    collate_fn=add_batch_padding,
)

model = params["model_class"](params)
#summary(model, input_size=tuple((next(iter(trainloader))[0]).shape))

optimizer = Adam(model.parameters(), lr=params["lr"])

trainer = Trainer(
    model=model,
    loss_fn=CrossEntropyLoss(),
    metrics=[MulticlassAccuracy()],
    optimizer=optimizer,
    early_stopping=EarlyStopping(10, 0.01, "min"),
    device="cpu",
    lrscheduler=ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5),
)

with mlflow.start_run():
    mlflow.log_params(params)
    trainer.fit(params["n_epochs"], trainloader, testloader)
    mlflow.pytorch.log_model(model, artifact_path="logged_models/model")
mlflow.end_run()
