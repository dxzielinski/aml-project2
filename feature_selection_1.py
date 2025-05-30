from data_loader import X_train, X_test, y_train, y_test
from data_loader import X, X_prediction, y
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

rs = 42


def mdi(X, y, t_variables):
    """
    Performs feature selection using Mean Decrease Impurity (MDI) from RandomForest.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Target vector of shape (n_samples,)
            t_variables (int): Number of top features to select

        Returns:
            np.ndarray: Indices of top t_variables features, sorted by importance
    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:t_variables]

    return top_indices


def final_score(accuracy, variables_num):
    """
    Computes the custom evaluation score for model selection.

        The scoring function reflects the business constraints:
        - 10,000 EUR * accuracy (reward for correct predictions)
        - 200 EUR * variables_num (cost for data acquisition)

        Args:
            accuracy (float): Model accuracy score [0-1]
            variables_num (int): Number of features used

        Returns:
            float: Final score combining accuracy and feature cost
    """
    return 10000 * accuracy - 200 * variables_num


def train_and_evaluate(t_values):
    """
    Main training pipeline that:
        1. Selects features using MDI
        2. Evaluates logistic regression models
        3. Saves model's predictions

        Returns:
            dict: Results for all tested configurations
    """

    results = []

    for t in range(1, t_values+1):
        for penalty in ['l1', 'l2']:
            for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:

                indices = mdi(X_train, y_train, t)

                lr = LogisticRegression(penalty=penalty, C=C, solver='liblinear', random_state=rs)
                lr.fit(X_train[:, indices], y_train)
                y_pred = lr.predict(X_test[:, indices])

                acc = accuracy_score(y_test, y_pred)
                score = final_score(acc, t)

                results.append({
                    't': t,
                    'penalty': penalty,
                    'C': C,
                    'accuracy': acc,
                    'score': score,
                    'selected_features': indices.tolist()
                })
    return results


def save_results(best_entry):
    """
    Generates output files using the best model configuration.

        Args:
        best_entry (dict): Contains:
            - 't': Number of features
            - 'penalty': Regularization type
            - 'C': Regularization strength
            - 'accuracy': Accuracy score
            - 'score': Final score with penalty
            - 'selected_features': Feature indices

        Creates:
        - vars1.txt: Selected feature indices
        - obs1.txt: Top 1000 predicted households
    """
    vars = mdi(X, y, best_entry['t'])

    lr = LogisticRegression(penalty=best_entry['penalty'], C=best_entry['C'], solver='liblinear', random_state=rs)
    lr.fit(X[:, vars], y)
    y_pred_proba = lr.predict_proba(X_prediction[:, vars])[:, 1]
    top_1000_indices = np.argsort(y_pred_proba)[::-1][:1000]

    with open('vars1.txt', 'w') as file:
        for num in vars:
            file.write(f"{num}\n")

    with open('obs1.txt', 'w') as file:
        for num in top_1000_indices:
            file.write(f"{num}\n")


if __name__ == '__main__':
    """Main execution block. For final, long training an optimal value is t_values=20"""
    results = train_and_evaluate(t_values=2)

    df = pd.DataFrame.from_dict(results)
    df.to_csv('results1.csv', index=False)

    best_entry = max(results, key=lambda x: x['score'])
    print("Best Model:", best_entry)
    save_results(best_entry)
