from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def create_pipeline(
    feature_engineering: str,
    n_neighbors: int,
    weights: str,
    leaf_size: int,
    n_jobs: int,
) -> Pipeline:
    pipeline_steps = []
    if feature_engineering == "PCA":
        pipeline_steps.append(("pca", PCA(n_components=30)))
    if feature_engineering == "Scaling":
        pipeline_steps.append(("scaler", MinMaxScaler()))
    pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                leaf_size=leaf_size,
                n_jobs=n_jobs,
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
