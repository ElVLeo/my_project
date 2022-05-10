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
    type=click.Choice(["PCA", "Scaling"]),
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
    default="gini",
    type=click.Choice(["gini", "entropy"]),
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
    n_estimators: int,
    criterion: str,
    max_depth: int,
    random_state: int,
) -> None:
    features, target = get_dataset(
        dataset_path,
    )
    with mlflow.start_run():
        model = create_pipeline(
            feature_engineering, n_estimators,
            criterion, max_depth, random_state
        )
        if grid_search:
            cv_inner = KFold(n_splits=5)
            space = {
                "classifier__n_estimators": [3, 8, 10],
                "classifier__criterion": ("gini", "entropy"),
                "classifier__max_depth": [None, 10, 20],
            }
            search = GridSearchCV(
                model, space, scoring="accuracy",
                n_jobs=1, cv=cv_inner, refit=True
            )
            model = search
            search.fit(features, target)
            parameters = search.best_params_
            n_estimators = parameters["classifier__n_estimators"]
            criterion = parameters["classifier__criterion"]
            max_depth = parameters["classifier__max_depth"]
        cv_outer = KFold(n_splits=5)
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
        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_param("feature_engineering", feature_engineering)
        mlflow.log_param("grid_search", grid_search)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision_macro: {precision_macro}.")
        click.echo(f"f1_weighted: {f1_weighted}.")
        dump(model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
