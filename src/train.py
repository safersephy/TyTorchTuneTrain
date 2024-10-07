import warnings

import mlflow
import mlflow.pytorch
from mads_datasets import DatasetFactoryProvider, DatasetType
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from tytorch.trainer import EarlyStopping, Trainer
from pathlib import Path

from tytorch.examples.datasets.flowers import FlowersDatasetFactory, ImgFactorySettings
from tytorch.data import TyTorchDataset

from models.image_classification import CNN
from utils.mlflow import get_training_config, set_mlflow_experiment

warnings.simplefilter("ignore", UserWarning)

params = get_training_config()
if params is None:
    params = {
        "model_class": CNN,
        "batch_size": 32,
        "n_epochs": 100,
        "input_size": (32, 3, 224, 224),
        "output_size": 5,
        "lr": 1e-4,
        "dropout": 0.3,
        "conv_blocks": [
            {"num_conv_layers": 3, "initial_filters": 32,"growth_factor": 2, "pool": True, "residual": False},  
            {"num_conv_layers": 3, "initial_filters": 256,"growth_factor": 1, "pool": False, "residual": True},
        ],
        "linear_blocks": [
            #{"out_features": 32, "dropout": 0.0},
            {"out_features": 16, "dropout": 0.0},
        ],
    }

#datasets = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS).create_dataset()

flowers_factory_settings  = ImgFactorySettings(
    source_url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    bronze_folder = Path("./tytorch/examples/data/bronze"), 
    bronze_filename="flowers.tgz",
    silver_folder = Path("./tytorch/examples/data/silver"),
    silver_filename="flowers.pt", 
    valid_frac=.2,
    test_frac=.2,
    unzip=True,
    formats=['.jpg','.png'],
    image_size=(224, 224)
)

train_dataset, valid_dataset, test_dataset = FlowersDatasetFactory(flowers_factory_settings
).load()




trainloader = DataLoader(
    train_dataset, batch_size=params["batch_size"], shuffle=True
)
testloader = DataLoader(
    valid_dataset, batch_size=params["batch_size"], shuffle=True
)

model = params["model_class"](params)
optimizer = Adam(model.parameters(), lr=params["lr"])

trainer = Trainer(
    model=model,
    loss_fn=CrossEntropyLoss(),
    metrics=[MulticlassAccuracy()],
    optimizer=optimizer,
    early_stopping=EarlyStopping(10, 0.01, "min"),
    device="mps",
    lrscheduler=ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5),
)

set_mlflow_experiment("train")
with mlflow.start_run():
    mlflow.log_params(params)
    trainer.fit(params["n_epochs"], trainloader, testloader)

    mlflow.pytorch.log_model(model, artifact_path="logged_models/model")
mlflow.end_run()
