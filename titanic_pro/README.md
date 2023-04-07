# Titanic Survival Prediction Pipeline
This repository contains a model training and inference pipeline for predicting the survival of passengers on the Titanic using the Titanic dataset. The pipeline utilizes Scikit-learn pipelines, custom transformers, XGBoost, grid search, cross-validation, feature selection, hyperparameter optimization with Optuna, Poetry for environment and library management, logging, and pytest for testing.

## Project Structure
The project is organized as follows:

```console
example_project/
├── data/
│   ├── train.csv
│   └── test.csv
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
└── pyproject.toml
```
## Setup
To set up the environment and install the required packages, follow these steps:

Install Poetry if you haven't already.

Clone the repository:

```console
git clone https://github.com/your_username/titanic-survival-prediction.git
cd titanic-survival-prediction
```
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

## Customization
You can customize this pipeline by modifying the following components:

example_pipeline/config.py: Change the feature and model configuration.
example_pipeline/pipeline.py: Update or extend the preprocessing pipeline.
example_pipeline/custom_transformers.py: Add new custom transformers for feature engineering.
example_pipeline/model.py: Modify the model or optimization process.
train.py and infer.py: Update the training and inference processes.
## License
This project is licensed under the MIT License.