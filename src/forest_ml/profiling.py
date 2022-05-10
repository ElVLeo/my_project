import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv("data/train.csv")


def profile():
    eda = ProfileReport(df, title="Pandas Profiling Report")
    eda.to_file(output_file="Pandas_profiling.html")
