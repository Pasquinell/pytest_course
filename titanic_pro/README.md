# Titanic Survival Prediction Pipeline
This repository contains a model training and inference pipeline for predicting the survival of passengers on the Titanic using the Titanic dataset. The pipeline utilizes Scikit-learn pipelines, custom transformers, XGBoost, grid search, cross-validation, feature selection, Poetry for environment and library management, logging, pytest for testing and pytest-cov for testing coverage checking. 
### TO-DO
- Add simple model versioning
- Add cvs mocking
## Warning
This repo was built for running in windows, so you shoulb be careful with pathing. You may want to change 
```python
dump(pipeline, 'titanic_pro/model.joblib')
```
in `train.py` and
```python
submission.to_csv("titanic_pro/submission.csv", index=False)
```
in `infer.py`.

## Project Structure
The project is organized as follows:

```console
titanic_pro/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── notebook1.ipynb
│   └── notebook2.ipynb
├── models/
│   ├── modelv1.joblib
│   └── modelv2.joblib
├── titanic_pro/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── model.py
│   ├── config.py
│   ├── custom_transformers.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_model.py
│   └── test_utils.py
├── train.py
├── infer.py
├── .gitignore.py
├── .coverage
└── pyproject.toml
```
## Setup
To set up the environment and install the required packages, follow these steps:

Install Poetry if you haven't already.

Clone the repository:

Install the dependencies:
```console
poetry install
```
Activate the virtual environment:
```console
poetry shell
```
## Usage
### Training
To train the model, run:

```console
python train.py
```
This script will load the Titanic dataset, preprocess the data, optimize the hyperparameters, perform feature selection, and train the XGBoost model using the best parameters.
### Warning
In windows I had to run from the parent directory calling titanic_pro and then train (the same for inference)!
```console
python .\titanic_pro\train.py
```
### Inference
To predict the survival of passengers in the test dataset, run:

```console
python infer.py
```
This script will load the trained model, preprocess the test data, and output the predictions.

## Testing
To run the test cases, execute:

```console
pytest
```
This will run the test cases in the tests/ directory.

## Testing coverage
For assesing the unit testing coverage run
```console
pytest --cov 
```
and then
```console
coverage combine
```

## Customization
You can customize this pipeline by modifying the following components:

- titanic_pro/config.py: Change the feature and model configuration.
- titanic_pro/pipeline.py: Update or extend the preprocessing pipeline.
- titanic_pro/custom_transformers.py: Add new custom transformers for feature engineering.
- titanic_pro/model.py: Modify the model or optimization process.
- train.py and infer.py: Update the training and inference processes.
## License
This project is licensed under the MIT License.