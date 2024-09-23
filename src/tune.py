import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from tytorch.trainer import Trainer
from tytorch.utils import get_device

from models.image_classification import cnn
from utils.mlflow import set_best_run_tag_and_log_model

warnings.simplefilter("ignore", UserWarning)
n_epochs = 15
n_trials = 50
batch_size = 32

tuningmetric = "valid_loss"
tuninggoal = "min"


search_space = {
    "model_class": cnn,
    "input_size": (batch_size, 3, 224, 224),  # Example: batch_size, channels, height, width
    "output_size": 5,  # Number of classes
    "lr": tune.loguniform(1e-5, 1e-3),        # Learning rate
    "dropout": tune.uniform(0.0,0.5),
    "num_conv_layers": 3,
    "initial_filters": 32,
    "linear_blocks": [
            {"out_features": 128, "dropout": 0.0},
            {"out_features": 64, "dropout": 0.0}
    ]
}

timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
experiment_path = f"tune-{timestamp}"

modelClass = search_space["model_class"]

datasetfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)


def tune_func(config):
    device = get_device()
    datasets = datasetfactory.create_dataset()
    trainloader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(datasets["valid"], batch_size=batch_size, shuffle=True)

    model = modelClass(config)
    model = model.to(device)

    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        metrics=[MulticlassAccuracy()],
        optimizer=Adam(model.parameters(), lr=config["lr"]),
        device=device,
    )

    mlflow.log_params(config)
    trainer.fit(n_epochs, trainloader, testloader)


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_path)

tuner = tune.Tuner(
    tune_func,
    param_space=search_space,
    tune_config=TuneConfig(
        mode=tuninggoal,
        scheduler=ASHAScheduler(),
        search_alg=HyperOptSearch(),
        metric=tuningmetric,
        num_samples=n_trials,
    ),
    run_config=train.RunConfig(
        storage_path=Path("./ray_tuning_results").resolve(),
        name=experiment_path,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                experiment_name=experiment_path,
                save_artifact=True,
            )
        ],
    ),
)
results = tuner.fit()

best_result = results.get_best_result(tuningmetric, tuninggoal)

model = modelClass(best_result.config)  # Initialize your model class
model.load_state_dict(
    torch.load(Path(best_result.checkpoint.path) / "model.pth")
)  # Load the saved weights
set_best_run_tag_and_log_model(experiment_path, model, tuningmetric, tuninggoal)

logger.info(f"tuning experiment name: {experiment_path}")
