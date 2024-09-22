from typing import Optional

import mlflow
from loguru import logger
from torch import nn


def set_best_run_tag_and_log_model(
    experiment_name: str, model: nn.Module, metric_name: str, direction: str = "max"
):
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
        print("No runs found in this experiment.")
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
        print(
            f"Best run with ID {best_run.info.run_id} has been tagged as the best run with a metric value of {best_metric}."
        )

        # Log the PyTorch model to the best run
        with mlflow.start_run(best_run.info.run_id):
            mlflow.pytorch.log_model(
                pytorch_model=model, artifact_path="logged_models/model"
            )
            print(f"Model has been logged to run {best_run.info.run_id}.")
    else:
        print(f"No valid runs found with the metric '{metric_name}'.")


def getModelfromMLFlow(
    experimentName, best_run: bool = False, runId: Optional[int] = None
):
    experiment = mlflow.get_experiment_by_name(experimentName)

    if experiment:
        experiment_id = experiment.experiment_id
    else:
        raise Exception(f"Experiment '{experimentName}' does not exist.")

    if runId is None:
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
            runId = last_run.run_id
            logger.info(f"Last run ID: {runId}")
            logger.info(f"Run details:\n{last_run}")
        else:
            logger.info(f"No runs found for experiment '{experimentName}'.")

    model_uri = f"runs:/{runId}/logged_models/model"
    param_uri = f"runs:/{runId}/params.json"
    param_dict = mlflow.artifacts.load_dict(param_uri)
    state_dict = mlflow.pytorch.load_model(model_uri).state_dict()
    return param_dict, state_dict
