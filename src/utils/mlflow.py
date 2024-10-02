import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
from loguru import logger
from torch import nn
from tytorch.utils import load_params_from_disk


def set_mlflow_experiment(experiment_name: str, add_timestamp: bool = True):
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_name = f"{experiment_name}-{timestamp}"

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    return experiment_name


def set_best_run_tag_and_log_model(
    experiment_name: str, model: nn.Module, metric_name: str, direction: str = "max"
) -> None:
    """
    Finds the run with the best metric result in an MLflow experiment (by name), sets a tag 'best_run=True',
    and logs a model to that run.

    Parameters:
    - experiment_name (str): The name of the MLflow experiment.
    - metric_name (str): The name of the metric to evaluate.
    - model_path (str): Path to the PyTorch model (.pth file) to log to the best run.
    - model_class (torch.nn.Module): The model class to load the saved state_dict into.
    - direction (str): The optimization direction, either 'min' or 'max'. Defaults to 'max'.

    """
    # Ensure that direction is valid
    if direction not in ["min", "max"]:
        raise ValueError("Direction must be either 'min' or 'max'")

    # Get experiment_id from experiment_name
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    experiment_id = experiment.experiment_id

    # Get all runs in the experiment
    runs = client.search_runs(experiment_id, order_by=[f"metrics.{metric_name}"])

    if not runs:
        logger.info("No runs found in this experiment.")
        return

    # Find the best run based on the direction
    best_run = None
    best_metric = None

    for run in runs:
        metric_value = run.data.metrics.get(metric_name)

        if metric_value is not None:
            if best_run is None:
                best_run = run
                best_metric = metric_value
            else:
                if (direction == "min" and metric_value < best_metric) or (
                    direction == "max" and metric_value > best_metric
                ):
                    best_run = run
                    best_metric = metric_value

    if best_run:
        # Set the tag 'best_run=True' for the best run
        client.set_tag(best_run.info.run_id, "best_run", "True")
        logger.info(
            f"Best run with ID {best_run.info.run_id} has been tagged as the best run with a metric value of {best_metric}."
        )

        # Log the PyTorch model to the best run
        with mlflow.start_run(best_run.info.run_id):
            mlflow.pytorch.log_model(
                pytorch_model=model, artifact_path="logged_models/model"
            )
            logger.info(
                f"Model has been logged to experimen {experiment_name}, run {best_run.info.run_id}."
            )
    else:
        logger.info(f"No valid runs found with the metric '{metric_name}'.")


def get_model_from_mlflow(
    experiment_name: str, best_run: bool = False, run_id: Optional[int] = None
) -> tuple[dict, dict]:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
    else:
        raise Exception(f"Experiment '{experiment_name}' does not exist.")

    if run_id is None:
        if best_run:
            # Filter runs with the tag "best_run" set to True
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string="tags.best_run = 'True'",
                order_by=["start_time DESC"],
            )
        else:
            # Get the most recent run
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id], order_by=["start_time DESC"]
            )

        if not runs.empty:
            last_run = runs.iloc[0]  # Get the most recent run
            run_id = last_run.run_id
            logger.info(f"Last run ID: {run_id}")
            logger.info(f"Run details:\n{last_run}")
        else:
            logger.info(f"No runs found for experiment '{experiment_name}'.")

    model_uri = f"runs:/{run_id}/logged_models/model"
    param_uri = f"runs:/{run_id}/params.json"
    param_dict = mlflow.artifacts.load_dict(param_uri)
    state_dict = mlflow.pytorch.load_model(model_uri).state_dict()
    return param_dict, state_dict


def get_training_config():
    parser = argparse.ArgumentParser(
        description="Train a model with the specified experiment name or config path"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment",
        default=None,  # Default value if not provided
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="path to config file",
        default=None,  # Default value if not provided
    )
    args = parser.parse_args()

    if args.experiment_name:
        print(f"Experiment name provided: {args.experiment_name}")
        experiment_name = args.experiment_name
        params, state = get_model_from_mlflow(experiment_name, True)

    elif args.config_path:
        print(f"config path provided: {args.config}")
        configfile = args.config_path
        params = load_params_from_disk(Path(configfile))

    else:
        print("No experiment name or config path provided. Set config manually")
        params = None
    return params
