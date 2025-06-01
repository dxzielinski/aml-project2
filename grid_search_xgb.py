import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import make_scorer, recall_score


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


class SelectTopKUnivariate(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that selects the top k features based on a
    univariate statistical test (ANOVA F-value).

    - On fit(X, y): computes the ANOVA F-scores for each feature, then retains
      the top k features (highest F-values).
    - On transform(X): returns X[:, selected_indices_].

    Parameters:
    -----------
    k : int
        Number of top features to select.
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        skb = SelectKBest(score_func=f_classif, k=self.k)
        skb.fit(X, y)
        mask = skb.get_support(indices=False)
        self.selected_indices_ = np.where(mask)[0]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def get_feature_names_out(self):
        return np.array(self.selected_indices_, dtype=object)


class SelectTopKRFE(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Recursive Feature Elimination (RFE) around a
    base estimator (by default RandomForestClassifier), and retains the top k features.

    - fit(X, y): runs RFE(base_estimator, n_features_to_select=k) on (X, y).
                 RFE ranks and selects k features.
    - transform(X): reduces X to only those k features.

    Hyperparameters:
    ---------------
    k : int
        Number of features to select.
    base_estimator : object
        Any estimator with a `feature_importances_` or `coef_`.
        Default = RandomForestClassifier(random_state=rs)
    step : int or float
        If int, number of features to remove at each iteration.
        If float, percentage (0<step<1) of features to remove at each iteration.
    """

    def __init__(self, k=5, step=0.1):
        self.k = k
        self.base_estimator = RandomForestClassifier(random_state=42)
        self.step = step

    def fit(self, X, y):
        """
        Fit an RFE selector with the given base_estimator to pick top k features.
        """
        rfe = RFE(
            estimator=self.base_estimator, n_features_to_select=self.k, step=self.step
        )
        rfe.fit(X, y)
        # rfe.support_ is a boolean mask of length n_features
        self.selected_indices_ = np.where(rfe.support_)[0]
        return self

    def transform(self, X):
        """
        Reduce X to only the selected k feature columns.
        """
        return X[:, self.selected_indices_]

    def get_feature_names_out(self):
        """
        Return the integer indices of the selected features.
        """
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
    proba = estimator.named_steps["xgb"].predict_proba(X_val_sel)[:, 1]
    n_val = len(proba)
    top_k = max(1, int(np.floor(0.2 * n_val)))
    top_indices = np.argsort(proba)[::-1][:top_k]
    tp = np.sum(y_val[top_indices] == 1)
    k = len(selected_idx)
    return 10 * int(tp) - 200 * k


if __name__ == "__main__":
    pipe = Pipeline(
        [
            ("selector", SelectTopK(k=5)),
            (
                "xgb",
                XGBClassifier(eval_metric="logloss", random_state=rs),
            ),
        ]
    )
    param_grid = {
        "selector__k": [2, 3, 4, 5],
        "xgb__max_depth": [1, 2, 3],
        "xgb__n_estimators": [1, 2, 3, 4, 5, 10, 20, 50, 100],
    }
    recall_scorer = make_scorer(
        recall_score, greater_is_better=True, pos_label=1, binary=True
    )

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=recall_scorer,
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
    k_best = best_params["selector__k"]
    selected_idx_test = best_pipeline.named_steps["selector"].selected_indices_
    X_test_sel = X_test[:, selected_idx_test]
    proba_test = best_pipeline.named_steps["xgb"].predict_proba(X_test_sel)[:, 1]
    top_k_test = 0.2 * len(proba_test)
    top_k_test = max(1, int(np.floor(top_k_test)))
    top_indices_test = np.argsort(proba_test)[::-1][:top_k_test]
    tp_test = np.sum(y_test[top_indices_test] == 1)
    business_test_score = 10 * int(tp_test) - 200 * k_best
    print(f"Test-set: Selected k = {k_best} features.")
    print(f"Test-set: True positives among top {top_k_test} = {tp_test}")
    print(f"Test-set business score = {business_test_score:.2f} €")
# Using SelectTopK with RandomForestClassifier for feature selection (the same for SelectTopKUnivariate):
# Best params found by GridSearchCV: {'selector__k': 1, 'xgb__max_depth': 2, 'xgb__n_estimators': 1}
# Best cross-validated business score: 1780.0
# Test-set: Selected k = 1 features.
# Test-set: True positives among top 1000 = 488
# Test-set business score = 4680.00 €
# RFE selection excluding k=1:
# Best params found by GridSearchCV: {'selector__k': 2, 'xgb__max_depth': 1, 'xgb__n_estimators': 10}
# Best cross-validated business score: 1603.3333333333333
# Test-set: Selected k = 2 features.
# Test-set: True positives among top 1000 = 488
# Test-set business score = 4480.00 €
