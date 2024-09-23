import argparse
import warnings
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.pytorch
from mads_datasets import DatasetFactoryProvider, DatasetType
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from tytorch.trainer import EarlyStopping, Trainer
from tytorch.utils import get_device, save_params_to_disk, load_params_from_disk

from models.image_classification import cnn, NeuralNetwork
from utils.mlflow import getModelfromMLFlow

warnings.simplefilter("ignore", UserWarning)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

parser = argparse.ArgumentParser(
    description="Train a model with the specified experiment name or param config"
)
parser.add_argument(
    "--experiment_name", 
    type=str, 
    help="Name of the experiment", 
    default=None  # Default value if not provided
)
parser.add_argument(
    "--config", 
    type=str, 
    help="name of the param file in the config folder", 
    default=None  # Default value if not provided
)
args = parser.parse_args()

if args.experiment_name:
    print(f"Experiment name provided: {args.experiment_name}")
    experiment_name = args.experiment_name
    params, state = getModelfromMLFlow(experiment_name, True)
    modelClass = cnn    
    datasetfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)    

elif args.config:
    print(f"param file provided: {args.config}")
    configfile = args.config
    params = load_params_from_disk(Path("./configs/params.pkl"))
    modelClass = params["model_class"]
    datasetfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)  

else:
    print("No experiment name provided. Using default settings.")

    params = {
        "model_class": cnn,
        "input_size": (32, 3, 224, 224),  # Example: batch_size, channels, height, width
        "output_size": 5,  # Number of classes
        "lr": 1e-3,        # Learning rate
        "conv_blocks": [   # Block-specific configurations
            {"filters": 32, "kernel_size": 3, "padding": 1, "dropout": 0.0, "maxpool": True},
            {"filters": 64, "kernel_size": 3, "padding": 0, "dropout": 0.0, "maxpool": True},
            {"filters": 128, "kernel_size": 3, "padding": 0, "dropout": 0.2, "maxpool": True},  
        ],
        "linear_blocks": [
        {"out_features": 128, "dropout": 0.0},
        {"out_features": 64, "dropout": 0.0}
        ]
    }
    modelClass = params["model_class"]
    datasetfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)       

batch_size = 32
n_epochs = 50


device = get_device()

timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
experiment_path = f"train-{timestamp}"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_path)


datasets = datasetfactory.create_dataset()
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
