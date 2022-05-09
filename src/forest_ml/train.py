from pathlib import Path
from joblib import dump
from numpy import mean

import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from .data import get_dataset
from .pipeline import create_pipeline


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
    type=click.Choice(['Scaling', 'Selecting']),
    show_default=True,
)
@click.option(
    "--n_estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default='gini',
    type=click.Choice(['gini', 'entropy']),
    show_default=True,
)
@click.option(
    "--max_depth",
    default=None,
    type=int,
    show_default=True,
)
@click.option(
    "--random_state",
    default=None,
    type=int,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    feature_engineering: str,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    random_state: int,
) -> None:
    features, target = get_dataset(
        dataset_path,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(feature_engineering, n_estimators, criterion, max_depth, random_state)
        cv = KFold(n_splits=10)
        accuracies = cross_val_score(pipeline, features, target, cv=cv, scoring='accuracy')
        accuracy = mean(accuracies)
        precision_micros = cross_val_score(pipeline, features, target, cv=cv, scoring='precision_micro')
        precision_micro = mean(precision_micros)
        f1s_weighted = cross_val_score(pipeline, features, target, cv=cv, scoring='f1_weighted')
        f1_weighted = mean(f1s_weighted)
        mlflow.log_param("model_name", 'RandomForestClassifier')
        mlflow.log_param("feature_engineering", feature_engineering)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_micro", precision_micro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision_micro: {precision_micro}.")
        click.echo(f"f1_weighted: {f1_weighted}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
