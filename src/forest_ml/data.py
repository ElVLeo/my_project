from pathlib import Path
from typing import Tuple

import click
import pandas as pd


def get_dataset(
    csv_path: Path
) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop(['Cover_Type', 'Id'], axis=1)
    click.echo(f"Features shape: {features.shape}.")
    target = dataset['Cover_Type']
    click.echo(f"Target shape: {target.shape}.")

    return features, target