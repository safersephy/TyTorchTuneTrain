import warnings
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from mads_datasets import DatasetFactoryProvider, DatasetType
from ray import train, tune
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from tytorch.trainer import Trainer
from torchvision import datasets, transforms
from models.image_classification import CNN
from utils.mlflow import set_best_run_tag_and_log_model, set_mlflow_experiment
from tytorch.examples.datasets.flowers import FlowersDatasetFactory, ImgFactorySettings

warnings.simplefilter("ignore", UserWarning)


tuningmetric = "valid_loss"
tuninggoal = "min"
n_trials = 60


params = {
    "model_class": CNN,
    "batch_size": 32,
    "n_epochs": 4,
    "device": "mps",
    "dataset_type": DatasetType.FLOWERS,
    "input_size": (32, 3, 224, 224),  # Example: batch_size, channels, height, width
    "output_size": 5,  # Number of classes
    "lr": 1e-4,  # Learning rate
    "dropout": .3,
    "conv_blocks": [
        {"num_conv_layers": tune.randint(1,8), "initial_filters": 32,"growth_factor": 2, "pool": True, "residual": True},  
        {"num_conv_layers": tune.randint(1,9), "initial_filters": 256,"growth_factor": 1, "pool": False, "residual": False},
    ],
    "linear_blocks": [
        #{"out_features": 32, "dropout": 0.0},
        {"out_features": 16, "dropout": 0.0},
    ],
}


experiment_name = set_mlflow_experiment("tune")

datasetfactory = DatasetFactoryProvider.create_factory(params["dataset_type"])
dataset = datasetfactory.create_dataset()

# from tytorch.examples.datasets.flowers import FlowersDatasetFactory, ImgFactorySettings

# flowers_factory_settings  = ImgFactorySettings(
#     source_url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
#     bronze_folder = Path.home() / "Development/TyTorch/TuneTrain/tytorch/examples/data/bronze", 
#     bronze_filename="flowers.tgz",
#     silver_folder = Path.home() / "Development/TyTorch/TuneTrain/tytorch/examples/data/silver",
#     silver_filename="flowers.pt", 
#     valid_frac=.2,
#     test_frac=.2,
#     unzip=True,
#     formats=['.jpg','.png'],
#     image_size=(224, 224)
# )



def tune_func(config):
    
    
    
    for idx, block in enumerate(config["conv_blocks"]):
        config[f"conv_block_{idx}_num_conv_layers"] = block["num_conv_layers"]
        config[f"conv_block_{idx}_initial_filters"] = block["initial_filters"]
        config[f"conv_block_{idx}_growth_factor"] = block["growth_factor"]
        config[f"conv_block_{idx}_pool"] = block["pool"]
        config[f"conv_block_{idx}_residual"] = block["residual"]
    
    dataset = datasetfactory.create_dataset()

    # train_dataset, valid_dataset, test_dataset = FlowersDatasetFactory(flowers_factory_settings
    #      ).create_datasets()
    
    trainloader = DataLoader(
       dataset["train"], batch_size=config["batch_size"], shuffle=True
    )
    testloader = DataLoader(
        dataset["valid"], batch_size=config["batch_size"], shuffle=True
    )

    model = config["model_class"](config)
    optimizer = Adam(model.parameters(), lr=params["lr"])
    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        metrics=[MulticlassAccuracy()],
        optimizer=optimizer,
        device=config["device"],
        lrscheduler=ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=5),
        quiet=True,
    )

    mlflow.log_params(config)


    
    trainer.fit(params["n_epochs"], trainloader, testloader)


tuner = tune.Tuner(
    tune.with_resources(tune_func, {"cpu": 10}),
    param_space=params,
    tune_config=TuneConfig(
        mode=tuninggoal,
        #scheduler=ASHAScheduler(),
        search_alg=HyperOptSearch(),
        metric=tuningmetric,
        num_samples=n_trials,
        max_concurrent_trials=1,

    ),
    run_config=train.RunConfig(
        storage_path=Path("./ray_tuning_results").resolve(),
        name=experiment_name,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                experiment_name=experiment_name,
                save_artifact=True,
            )
        ],

    ),
)
results = tuner.fit()

best_result = results.get_best_result(tuningmetric, tuninggoal)
model = params["model_class"](best_result.config)
model.load_state_dict(torch.load(Path(best_result.checkpoint.path) / "model.pth"))
set_best_run_tag_and_log_model(experiment_name, model, tuningmetric, tuninggoal)
