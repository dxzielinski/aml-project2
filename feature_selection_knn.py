from data_loader import X_train, X_test, y_train, y_test, X_prediction
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

rs = 42

print("Start KNN pipeline")

def anova_selection(X, y, t_variables):
    """
    Selects top t_variables using ANOVA F-test
    """
    selector = SelectKBest(score_func=f_classif, k=t_variables)
    selector.fit(X, y)
    selected = selector.get_support(indices=True)
    return selected

def final_score(accuracy, variables_num):
    return 10000 * accuracy - 200 * variables_num

def train_and_evaluate(t_values):
    results = []

    for t in range(1, t_values + 1):
        for k in [3, 5, 7, 11]:
            indices = anova_selection(X_train, y_train, t)

            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train[:, indices], y_train)
            y_pred = clf.predict(X_test[:, indices])

            acc = accuracy_score(y_test, y_pred)
            score = final_score(acc, t)

            results.append(
                {
                    "t": t,
                    "n_neighbors": k,
                    "accuracy": acc,
                    "score": score,
                    "selected_features": indices.tolist(),
                }
            )
    return results

def save_results(best_entry):
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
    results = train_and_evaluate(t_values=2)
    df = pd.DataFrame(results)
    df.to_csv("results_knn.csv", index=False)
    best_entry = max(results, key=lambda x: x["score"])
    print("Best KNN model:", best_entry)
    save_results(best_entry)

print("success")
