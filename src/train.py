import argparse
import warnings
from datetime import datetime

import mlflow
import mlflow.pytorch
from mads_datasets import DatasetFactoryProvider, DatasetType
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from tytorch.trainer import EarlyStopping, Trainer
from tytorch.utils import get_device

from models.image_classification import cnn
from utils.mlflow import getModelfromMLFlow

warnings.simplefilter("ignore", UserWarning)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

parser = argparse.ArgumentParser(
    description="Train a model with the specified experiment name"
)
parser.add_argument("experiment_name", type=str, help="Name of the experiment")
args = parser.parse_args()
experiment_name = args.experiment_name

batch_size = 32
n_epochs = 2
modelClass = cnn

device = get_device()

params, state = getModelfromMLFlow(experiment_name, True)

timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:S")
experiment_path = f"train-{timestamp}"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_path)

fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
datasets = fashionfactory.create_dataset()
trainloader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
testloader = DataLoader(datasets["valid"], batch_size=batch_size, shuffle=True)

model = modelClass(params).to(device)
trainer = Trainer(
    model=model,
    loss_fn=CrossEntropyLoss(),
    metrics=[MulticlassAccuracy()],
    optimizer=Adam(model.parameters(), lr=params["lr"]),
    early_stopping=EarlyStopping(5, 0.01, "min"),
    device=device,
)

with mlflow.start_run():
    mlflow.log_params(params)
    trainer.fit(n_epochs, trainloader, testloader)
    mlflow.pytorch.log_model(model, artifact_path="logged_models/model")
mlflow.end_run()
