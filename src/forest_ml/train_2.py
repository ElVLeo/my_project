from pathlib import Path
from joblib import dump
from numpy import mean

import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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
    "--feature_engineering",
    default=None,
    type=click.Choice(["PCA", "Scaling"]),
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
    default="uniform",
    type=click.Choice(["uniform", "distance"]),
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
@click.option(
    "--grid_search",
    default=True,
    type=bool,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    grid_search: bool,
    feature_engineering: str,
    n_neighbors: int,
    weights: str,
    leaf_size: int,
    n_jobs: int,
) -> None:
    features, target = get_dataset(
        dataset_path,
    )
    with mlflow.start_run():
        model = create_pipeline(
            feature_engineering, n_neighbors, weights, leaf_size, n_jobs
        )
        if grid_search:
            cv_inner = KFold(n_splits=3)
            space = {
                "classifier__n_neighbors": [5, 10],
                "classifier__weights": ("uniform", "distance"),
                "classifier__leaf_size": [5, 10],
                "classifier__n_jobs": [None, -1],
            }
            search = GridSearchCV(
                model, space, scoring="accuracy", n_jobs=1, cv=cv_inner, refit=True
            )
            model = search
            search.fit(features, target)
            parameters = search.best_params_
            n_neighbors = parameters["classifier__n_neighbors"]
            weights = parameters["classifier__weights"]
            leaf_size = parameters["classifier__leaf_size"]
            n_jobs = parameters["classifier__n_jobs"]
        cv_outer = KFold(n_splits=3)
        accuracies = cross_val_score(
            model, features, target, cv=cv_outer, scoring="accuracy"
        )
        accuracy = mean(accuracies)
        precision_macros = cross_val_score(
            model, features, target, cv=cv_outer, scoring="precision_macro"
        )
        precision_macro = mean(precision_macros)
        f1s_weighted = cross_val_score(
            model, features, target, cv=cv_outer, scoring="f1_weighted"
        )
        f1_weighted = mean(f1s_weighted)
        mlflow.log_param("model_name", "KNeighborsClassifier")
        mlflow.log_param("feature_engineering", feature_engineering)
        mlflow.log_param("grid_search", grid_search)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("weights", weights)
        mlflow.log_param("leaf_size", leaf_size)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision_macro: {precision_macro}.")
        click.echo(f"f1_weighted: {f1_weighted}.")
        dump(model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
