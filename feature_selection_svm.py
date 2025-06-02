from data_loader import X_train, X_test, y_train, y_test, X_prediction, percent_correct_among_top20percent
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

rs = 42

print("Start SVM pipeline")


def l1_feature_selection(X, y, t_variables):
    """
    Selects top t_variables using L1-penalized Logistic Regression.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        t_variables (int): Number of features to select

    Returns:
        np.ndarray: Indices of selected features
    """
    model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=rs)
    model.fit(X, y)
    coef = np.abs(model.coef_)[0]
    top_indices = np.argsort(coef)[::-1][:t_variables]
    return top_indices


def final_score(accuracy, variables_num):
    """
    Computes the custom score with business constraint tradeoffs.

    Args:
        accuracy (float): Accuracy among top 20% predictions
        variables_num (int): Number of features used

    Returns:
        float: Final evaluation score
    """
    return 10000 * accuracy - 200 * variables_num


def train_and_evaluate(t_values):
    """
    Trains SVM models with L1-based feature selection and evaluates them.

    Args:
        t_values (int): Maximum number of features to test (1 to t_values)

    Returns:
        list[dict]: List of results with hyperparameters, performance, and selected features
    """
    results = []

    for t in range(1, t_values + 1):
        for kernel in ["linear", "rbf"]:
            for C in [0.1, 1, 10]:
                selected_features = l1_feature_selection(X_train, y_train, t)

                clf = SVC(C=C, kernel=kernel, probability=True, random_state=rs)
                clf.fit(X_train[:, selected_features], y_train)
                y_pred = clf.predict(X_test[:, selected_features])

                acc = accuracy_score(y_test, y_pred)
                correct_percent = percent_correct_among_top20percent(clf, X_test[:, selected_features], y_test)
                score = final_score(correct_percent, t)

                results.append(
                    {
                        "t": t,
                        "kernel": kernel,
                        "C": C,
                        "accuracy": acc,
                        "score_among_top_20_percent": score,
                        "selected_features": selected_features.tolist(),
                    }
                )
    return results


def save_results(best_entry):
    """
    Saves selected features and top predicted observations using the best model.

    Args:
        best_entry (dict): Best model configuration and results
    """
    selected_features = l1_feature_selection(X_train, y_train, best_entry["t"])

    clf = SVC(C=best_entry["C"], kernel=best_entry["kernel"], probability=True, random_state=rs)
    clf.fit(X_train[:, selected_features], y_train)
    y_pred_proba = clf.predict_proba(X_prediction[:, selected_features])[:, 1]
    top_1000_indices = np.argsort(y_pred_proba)[::-1][:1000]

    with open("vars_svm.txt", "w") as f:
        for v in selected_features:
            f.write(f"{v}\n")

    with open("obs_svm.txt", "w") as f:
        for idx in top_1000_indices:
            f.write(f"{idx}\n")


if __name__ == "__main__":
    results = train_and_evaluate(t_values=20)
    df = pd.DataFrame(results)
    df.to_csv("results_svm.csv", index=False)
    best_entry = max(results, key=lambda x: x["score_among_top_20_percent"])
    print("Best SVM model:", best_entry)
    save_results(best_entry)

print("Success")
