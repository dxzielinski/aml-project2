from data_loader import X_train, X_test, y_train, y_test, X_prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

rs = 42

print("start RFE + RF pipeline")

def rfe_selection(X, y, t_variables):
    """
    Perform feature selection using Recursive Feature Elimination (RFE)
    with a Random Forest base estimator.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=rs)
    selector = RFE(estimator=rf, n_features_to_select=t_variables, step=0.1)
    selector.fit(X, y)
    return np.where(selector.support_)[0]

def final_score(accuracy, variables_num):
    return 10000 * accuracy - 200 * variables_num

def train_and_evaluate(t_values):
    results = []

    for t in range(1, t_values + 1):
        for n_estimators in [50, 100, 150]:
            indices = rfe_selection(X_train, y_train, t)

            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=rs)
            clf.fit(X_train[:, indices], y_train)
            y_pred = clf.predict(X_test[:, indices])

            acc = accuracy_score(y_test, y_pred)
            score = final_score(acc, t)

            results.append(
                {
                    "t": t,
                    "n_estimators": n_estimators,
                    "accuracy": acc,
                    "score": score,
                    "selected_features": indices.tolist(),
                }
            )
    return results

def save_results(best_entry):
    vars = rfe_selection(X_train, y_train, best_entry["t"])

    clf = RandomForestClassifier(n_estimators=best_entry["n_estimators"], random_state=rs)
    clf.fit(X_train[:, vars], y_train)
    y_pred_proba = clf.predict_proba(X_prediction[:, vars])[:, 1]
    top_1000_indices = np.argsort(y_pred_proba)[::-1][:1000]

    with open("vars_rfe_rf.txt", "w") as f:
        for v in vars:
            f.write(f"{v}\n")

    with open("obs_rfe_rf.txt", "w") as f:
        for idx in top_1000_indices:
            f.write(f"{idx}\n")

if __name__ == "__main__":
    results = train_and_evaluate(t_values=2)
    df = pd.DataFrame(results)
    df.to_csv("results_rfe_rf.csv", index=False)
    best_entry = max(results, key=lambda x: x["score"])
    print("Best RFE + RF model:", best_entry)
    save_results(best_entry)

print("Success")
