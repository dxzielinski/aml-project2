import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

from data_loader import X_train, X_test, y_train, y_test

rs = 42


class SelectTopK(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that:
      - On fit(X, y): fits a RandomForestClassifier to (X, y) and finds the top k
        features by MDI (mean decrease impurity).
      - On transform(X): returns X[:, selected_indices_]

    We'll expose a parameter `k` so that GridSearchCV can try k=1..t_max.
    """

    def __init__(self, k=5, rf_random_state=rs):
        self.k = k
        self.rf_random_state = rf_random_state

    def fit(self, X, y):
        rf = RandomForestClassifier(random_state=self.rf_random_state)
        rf.fit(X, y)
        importances = rf.feature_importances_
        self.selected_indices_ = np.argsort(importances)[::-1][: self.k]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def get_feature_names_out(self):
        return np.array(self.selected_indices_, dtype=object)


def business_scorer(estimator, X_val, y_val):
    """
    Custom scoring function for XGB + feature-selection pipeline.
    - estimator: a fitted Pipeline([('selector', SelectTopK()), ('xgb', XGBClassifier())])
    - X_val:    2D array of held-out samples
    - y_val:    1D array of true labels for held-out

    Returns:
      business_score = 10 * (true positives on X_val) - 200 * (k)

    Note: this must return a scalar. GridSearchCV will maximize it.
    """
    selected_idx = estimator.named_steps["selector"].selected_indices_
    X_val_sel = X_val[:, selected_idx]
    y_pred = estimator.named_steps["xgb"].predict(X_val_sel)
    tp = np.sum((y_val == 1) & (y_pred == 1))
    k = len(selected_idx)
    return 10 * tp - 200 * k


if __name__ == "__main__":
    pipe = Pipeline(
        [
            ("selector", SelectTopK(k=5, rf_random_state=rs)),
            (
                "xgb",
                XGBClassifier(eval_metric="logloss", random_state=rs),
            ),
        ]
    )
    param_grid = {
        "selector__k": [1, 2, 5, 10, 20],
        "xgb__max_depth": [2, 3, 4, 5],
        "xgb__learning_rate": [0.01, 0.1],
        "xgb__n_estimators": [100, 200, 300],
    }
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=business_scorer,
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best params found by GridSearchCV:", best_params)
    print("Best cross-validated business score:", best_score)
    best_pipeline = grid_search.best_estimator_
    y_test_pred = best_pipeline.predict(X_test)
    test_acc = np.mean(y_test_pred == y_test)
    tp_test = np.sum((y_test == 1) & (y_test_pred == 1))
    k_best = best_params["k"]
    business_test_score = 10 * tp_test - 200 * k_best
    print(f"Test-set accuracy: {test_acc:.4f}")
    print(f"Test-set business score: {business_test_score:.2f} €")
