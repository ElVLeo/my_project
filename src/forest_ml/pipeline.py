from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def create_pipeline(
    feature_engineering: str, n_estimators: int, criterion: str, max_depth: int, random_state: int,
) -> Pipeline:
    pipeline_steps = []
    if feature_engineering == 'PCA':
        pipeline_steps.append(("pca", PCA(n_components=30)))
    if feature_engineering == 'Scaling':
        pipeline_steps.append(("scaler", MinMaxScaler()))
    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=random_state
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
