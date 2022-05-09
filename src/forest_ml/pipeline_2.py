from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    feature_engineering: str, n_neighbors: int, weights: str, leaf_size: int, n_jobs: int
) -> Pipeline:
    pipeline_steps = []
    if feature_engineering == 'Scaling':
        pipeline_steps.append(("scaler", StandardScaler()))
    if feature_engineering == 'Selecting':
        pipeline_steps.append(("selector", SelectKBest(chi2, k=10)))
    pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, n_jobs=n_jobs
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
