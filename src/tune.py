from pathlib import Path
from mads_datasets import DatasetFactoryProvider, DatasetType
import warnings
import mlflow
from loguru import logger
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torcheval.metrics import MulticlassAccuracy
from tytorch.trainer import Trainer
from tytorch.utils import get_device
from utils.mlflow import set_best_run_tag_and_log_model
from models.image_classification import cnn
from datetime import datetime
import mlflow.pytorch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import TuneConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback

warnings.simplefilter("ignore", UserWarning)
n_epochs = 2
n_trials = 2
batch_size = 32

tuningmetric = "valid_loss"
tuninggoal = "min"

search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "filters": tune.qrandint(32, 128, 16),
    "h1": tune.qrandint(32, 256, 16),
    "h2": tune.qrandint(16, 128, 16),
    "input_size": (batch_size, 1, 28, 28),
}

timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
experiment_path = f"tune-{timestamp}"

modelClass = cnn

datasetfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)


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
