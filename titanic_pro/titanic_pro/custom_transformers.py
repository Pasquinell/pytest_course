import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        age_filled = X.copy()
        age_filled["age"].fillna(age_filled["age"].median(), inplace=True)
        return age_filled
