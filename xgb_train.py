import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from data_loader import X_train, X_test, X_prediction, y_train, y_test
from feature_selection_1 import mdi, final_score

rs = 42


def train_and_evaluate_xgb(t_values):
    """
    Main training pipeline that:
        1. Selects features using MDI
        2. Evaluates XGBoost models (sweeping over a small hyperparam grid)
        3. Returns results for all tested configurations

    Args:
        t_values (int): Maximum number of features to test (from 1 to t_values)

    Returns:
        list of dict: Each dict contains:
            - 't': number of features used
            - 'max_depth': XGBoost max tree depth
            - 'learning_rate': XGBoost learning rate (eta)
            - 'n_estimators': number of boosting rounds
            - 'accuracy': accuracy on X_test
            - 'score': final_score (10000*accuracy - 200*t)
            - 'selected_features': list of feature indices
    """
    results = []

    max_depth_list = [2, 3]
    n_estimators_list = [100, 200]

    for t in range(1, t_values + 1):
        indices = mdi(X_train, y_train, t)

        for max_depth in max_depth_list:
            for n_estimators in n_estimators_list:
                xgb = XGBClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    eval_metric="logloss",
                    random_state=rs,
                )

                xgb.fit(X_train[:, indices], y_train)

                y_pred = xgb.predict(X_test[:, indices])

                acc = accuracy_score(y_test, y_pred)
                score = final_score(acc, t)

                results.append(
                    {
                        "t": t,
                        "max_depth": max_depth,
                        "n_estimators": n_estimators,
                        "accuracy": acc,
                        "score": score,
                        "selected_features": indices.tolist(),
                    }
                )

    return results


def save_results_xgb(best_entry):
    """
    Generates output files using the best XGBoost configuration.

        Args:
            best_entry (dict): Contains:
                - 't': number of features
                - 'max_depth': XGBoost max tree depth
                - 'learning_rate': learning rate
                - 'n_estimators': number of boosting rounds
                - 'accuracy': accuracy score
                - 'score': final score with penalty
                - 'selected_features': feature indices

        Creates:
            - vars_xgb.txt: Selected feature indices
            - obs_xgb.txt: Top 1000 predicted households (by probability)
    """
    t = best_entry["t"]
    max_depth = best_entry["max_depth"]
    n_estimators = best_entry["n_estimators"]

    vars_indices = mdi(X_train, y_train, t)

    xgb_final = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        eval_metric="logloss",
        random_state=rs,
    )
    xgb_final.fit(X_train[:, vars_indices], y_train)

    y_pred_proba = xgb_final.predict_proba(X_prediction[:, vars_indices])[:, 1]
    top_1000_indices = np.argsort(y_pred_proba)[::-1][:1000]

    with open("vars_xgb.txt", "w") as file:
        for idx in vars_indices:
            file.write(f"{idx}\n")

    with open("obs_xgb.txt", "w") as file:
        for idx in top_1000_indices:
            file.write(f"{idx}\n")


if __name__ == "__main__":
    results_xgb = train_and_evaluate_xgb(t_values=20)
    df_xgb = pd.DataFrame.from_dict(results_xgb)
    df_xgb.to_csv("results_xgb.csv", index=False)
    best_entry_xgb = max(results_xgb, key=lambda x: x["score"])
    print("Best XGBoost Model:", best_entry_xgb)
    save_results_xgb(best_entry_xgb)
