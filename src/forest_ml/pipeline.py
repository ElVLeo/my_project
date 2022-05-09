from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_pipeline(
    use_scaler: str, n_estimators: int, criterion: str, max_depth: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler == 'StandardScaling':
        pipeline_steps.append(("scaler", StandardScaler()))
    else:
        pipeline_steps.append(("scaler", MinMaxScaler()))
    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators, criterion=criterion, max_depth=max_depth
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
