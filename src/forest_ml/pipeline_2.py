from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_pipeline(
    use_scaler: str, n_neighbors: int, weights: str, leaf_size: int, n_jobs: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler == 'StandardScaling':
        pipeline_steps.append(("scaler", StandardScaler()))
    else:
        pipeline_steps.append(("scaler", MinMaxScaler()))
    pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, n_jobs=n_jobs
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
