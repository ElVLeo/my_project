## Usage
This package allows you to train model for forest cover type prediction.
1. Clone this repository to your machine.
2. Download [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction), save csv locally (default path is *data/heart.csv* in repository's root).
3. Make sure Python 3.9.0 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command (if you want to use model RandomForestClassifier):
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run train_2 with the following command (if you want to use model KNeighborsClassifier):
```sh
poetry run train_2 -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train_2 --help
```
7. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
8. Run eda to form and see EDA (pandas-profiling in html report):
```sh
poetry run eda
```
## Development

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox [-r]
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
nox -[r]s black
poetry run black src tests noxfile.py
```
Format your code with [flake8](https://github.com/psf/black) by using either nox or poetry:
```
nox -[r]s flake8
poetry run flake8 src tests noxfile.py
```

## My results

Task 8. You also can get this metrics if you use `grid_search=False` and add necessary metrics to every model. If you want to have representative results you should use random_state=2022 for train (RandomForestClassifier). For example,  
```
poetry run train --grid_search=False --feature_engineering='Scaling' --n_estimators=100 --criterion='entropy' --max_depth=40 --random_state=2022
```


I chose accuracy like a metric for choosing the best model. You can see this metrics and parameters on the screen above. This model is the first.

Task 9. You can see on the screen the best parameters with metrics after NestedCV for every model. Sorry, I can't use a lot of parameters for GridSearchCV (because my computer not so good as I want), but I could write a script for this and learn how it works. 
![Image text](https://github.com/ElVLeo/machine_learning/blob/main/%D1%81%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D0%B4%D0%BB%D1%8F%20readme.PNG)

Task 12. A screenshot that linting and formatting are passed on example src
![img_3.png](img_3.png)
![img_4.png](img_4.png)
