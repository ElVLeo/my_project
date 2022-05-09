from pathlib import Path
from joblib import dump
from numpy import mean

import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from .data import get_dataset
from .pipeline_2 import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default='StandardScaling',
    type=click.Choice(['StandardScaling', 'MinMaxScaling']),
    show_default=True,
)
@click.option(
    "--n_neighbors",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--weights",
    default='uniform',
    type=click.Choice(['uniform', 'distance']),
    show_default=True,
)
@click.option(
    "--leaf_size",
    default=30,
    type=int,
    show_default=True,
)
@click.option(
    "--n_jobs",
    default=None,
    type=int,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    use_scaler: str,
    n_neighbors: int,
    weights: str,
    leaf_size: int,
    n_jobs: int,
) -> None:
    features, target = get_dataset(
        dataset_path,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, n_neighbors, weights, leaf_size, n_jobs)
        cv = KFold(n_splits=10)
        accuracies = cross_val_score(pipeline, features, target, cv=cv, scoring='accuracy')
        accuracy = mean(accuracies)
        precision_micros = cross_val_score(pipeline, features, target, cv=cv, scoring='precision_micro')
        precision_micro = mean(precision_micros)
        f1s_weighted = cross_val_score(pipeline, features, target, cv=cv, scoring='f1_weighted')
        f1_weighted = mean(f1s_weighted)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("weights", weights)
        mlflow.log_param("leaf_size", leaf_size)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_micro", precision_micro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision_micro: {precision_micro}.")
        click.echo(f"f1_weighted: {f1_weighted}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
