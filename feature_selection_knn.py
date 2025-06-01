from data_loader import X_train, X_test, y_train, y_test, X_prediction, percent_correct_among_top20percent
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

rs = 42

print("Start KNN pipeline")


def anova_selection(X, y, t_variables):
    """
    Selects top t_variables using ANOVA F-test.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        t_variables (int): Number of top features to select

    Returns:
        np.ndarray: Indices of selected features
    """
    selector = SelectKBest(score_func=f_classif, k=t_variables)
    selector.fit(X, y)
    selected = selector.get_support(indices=True)
    return selected


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
    Trains KNN models with ANOVA-selected features and evaluates them.

    Args:
        t_values (int): Maximum number of features to test (1 to t_values)

    Returns:
        list[dict]: List of results with hyperparameters, performance, and selected features
    """
    results = []

    for t in range(1, t_values + 1):
        for k in [3, 5, 7, 11]:
            indices = anova_selection(X_train, y_train, t)

            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train[:, indices], y_train)
            y_pred = clf.predict(X_test[:, indices])

            correct_percent = percent_correct_among_top20percent(clf, X_test[:, indices], y_test)
            acc = accuracy_score(y_test, y_pred)
            score = final_score(correct_percent, t)

            results.append(
                {
                    "t": t,
                    "n_neighbors": k,
                    "accuracy": acc,
                    "score_among_top_20_percent": score,
                    "selected_features": indices.tolist(),
                }
            )
    return results


def save_results(best_entry):
    """
    Saves selected features and top predicted observations using the best model.

    Args:
        best_entry (dict): Best model configuration and results
    """
    vars = anova_selection(X_train, y_train, best_entry["t"])

    clf = KNeighborsClassifier(n_neighbors=best_entry["n_neighbors"])
    clf.fit(X_train[:, vars], y_train)
    y_pred_proba = clf.predict_proba(X_prediction[:, vars])[:, 1]
    top_1000_indices = np.argsort(y_pred_proba)[::-1][:1000]

    with open("vars_knn.txt", "w") as f:
        for v in vars:
            f.write(f"{v}\n")

    with open("obs_knn.txt", "w") as f:
        for idx in top_1000_indices:
            f.write(f"{idx}\n")


if __name__ == "__main__":
    results = train_and_evaluate(t_values=20)  # Adjust t_values as needed
    df = pd.DataFrame(results)
    df.to_csv("results_knn.csv", index=False)
    best_entry = max(results, key=lambda x: x["score_among_top_20_percent"])
    print("Best KNN model:", best_entry)
    save_results(best_entry)

print("Success")
