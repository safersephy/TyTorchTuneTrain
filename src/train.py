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
from torchinfo import summary
from models.image_classification import CNN, NeuralNetwork
from utils.mlflow import set_mlflow_experiment, get_training_config
warnings.simplefilter("ignore", UserWarning)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
set_mlflow_experiment("train")

params = get_training_config()
if params is None:
    params = {
        "model_class": CNN,
        "batch_size": 32,
        "n_epochs": 50,
        "device":"cpu",
        "dataset_type": DatasetType.FLOWERS,
        "input_size": (32, 3, 224, 224),  
        "output_size": 5,  
        "lr": 1e-4,        
        "dropout": 0.3,
        "num_conv_layers": 5,
        "initial_filters": 32,
        "linear_blocks": [
        {"out_features": 32, "dropout": 0.0},
        {"out_features": 16, "dropout": 0.0}
        ]
    }
datasets = DatasetFactoryProvider.create_factory(params["dataset_type"]).create_dataset()       
trainloader = DataLoader(datasets["train"], batch_size=params["batch_size"], shuffle=True)
testloader = DataLoader(datasets["valid"], batch_size=params["batch_size"], shuffle=True)

model = params["model_class"](params)
summary(model, input_size=tuple((next(iter(trainloader))[0]).shape))

optimizer = Adam(model.parameters(), lr=params["lr"])

trainer = Trainer(
    model=model,
    loss_fn=CrossEntropyLoss(),
    metrics=[MulticlassAccuracy()],
    optimizer=optimizer,
    early_stopping=EarlyStopping(10, 0.01, "min"),
    device=params["device"],
    lrscheduler=ReduceLROnPlateau(optimizer=optimizer, factor=.5, patience=5)
)

with mlflow.start_run():
    mlflow.log_params(params)
    trainer.fit(params["n_epochs"], trainloader, testloader)
    
    mlflow.pytorch.log_model(model, artifact_path="logged_models/model")
mlflow.end_run()
