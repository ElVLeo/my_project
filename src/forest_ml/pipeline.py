from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    feature_engineering: str, n_estimators: int, criterion: str, max_depth: int
) -> Pipeline:
    pipeline_steps = []
    if feature_engineering == 'Scaling':
        pipeline_steps.append(("scaler", StandardScaler()))
    if feature_engineering == 'Selecting':
        pipeline_steps.append(("selector", SelectKBest(chi2, k=10)))
    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators, criterion=criterion, max_depth=max_depth
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
