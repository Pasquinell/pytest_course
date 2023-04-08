from sklearn.model_selection import GridSearchCV
from .pipeline import pipeline


def tune_model(X, y):
    grid_search_params = {
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
        "classifier__n_estimators": [50, 100, 200],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=grid_search_params,
        scoring="accuracy",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_params, best_score
