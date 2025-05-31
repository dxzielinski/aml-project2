from data_loader import X_train, X_test, y_train, y_test, X_prediction
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

rs = 42

print("Start SVM")

def l1_feature_selection(X, y, t_variables):
    """
    Select top t_variables using L1-penalized Logistic Regression.
    """
    model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=rs)
    model.fit(X, y)
    coef = np.abs(model.coef_)[0]
    top_indices = np.argsort(coef)[::-1][:t_variables]
    return top_indices

def final_score(accuracy, variables_num):
    return 10000 * accuracy - 200 * variables_num

def train_and_evaluate(t_values):
    results = []

    for t in range(1, t_values + 1):
        for kernel in ["linear", "rbf"]:
            for C in [0.1, 1, 10]:
                indices = l1_feature_selection(X_train, y_train, t)

                clf = SVC(C=C, kernel=kernel, probability=True, random_state=rs)
                clf.fit(X_train[:, indices], y_train)
                y_pred = clf.predict(X_test[:, indices])

                acc = accuracy_score(y_test, y_pred)
                score = final_score(acc, t)

                results.append(
                    {
                        "t": t,
                        "kernel": kernel,
                        "C": C,
                        "accuracy": acc,
                        "score": score,
                        "selected_features": indices.tolist(),
                    }
                )
    return results

def save_results(best_entry):
    vars = l1_feature_selection(X_train, y_train, best_entry["t"])

    clf = SVC(C=best_entry["C"], kernel=best_entry["kernel"], probability=True, random_state=rs)
    clf.fit(X_train[:, vars], y_train)
    y_pred_proba = clf.predict_proba(X_prediction[:, vars])[:, 1]
    top_1000_indices = np.argsort(y_pred_proba)[::-1][:1000]

    with open("vars_svm.txt", "w") as f:
        for v in vars:
            f.write(f"{v}\n")

    with open("obs_svm.txt", "w") as f:
        for idx in top_1000_indices:
            f.write(f"{idx}\n")

if __name__ == "__main__":
    results = train_and_evaluate(t_values=2)
    df = pd.DataFrame(results)
    df.to_csv("results_svm.csv", index=False)
    best_entry = max(results, key=lambda x: x["score"])
    print("Best SVM model:", best_entry)
    save_results(best_entry)

print("Success")
