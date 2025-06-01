import numpy as np
from data_loader import X_train, y_train
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def cohend_not_abs(mu_0, mu_1, std_0, std_1, n_0, n_1):
    """
    Calculates Cohen's d effect size without taking absolute value.

        This computes the standardized mean difference between two groups, keeping
        the directionality (sign) of the effect.

        Args:
            mu_0: Mean of group 0
            mu_1: Mean of group 1
            std_0: Standard deviation of group 0
            std_1: Standard deviation of group 1
            n_0: Sample size of group 0
            n_1: Sample size of group 1

        Returns:
            float: Cohen's d without absolute value (positive values indicate
            group 1 > group 0)
    """
    sigma = np.sqrt(((n_0 - 1) * (std_0 ** 2) + (n_1 - 1) * (std_1 ** 2)) / (n_0 + n_1 - 2))
    d_not_abs = (mu_1 - mu_0) / sigma
    return d_not_abs


def separability_dataframe(X, y):
    """
    Calculates and ranks feature separability using Cohen's d metric.

        For each feature in the input data, computes the Cohen's d effect size
        between classes to measure how well the feature separates the classes.
        Returns a sorted dataframe of features ranked by their separability.

        Args:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
            y (numpy.ndarray): Target labels of shape (n_samples,)

        Returns:
            pandas.DataFrame: Sorted dataframe containing:
                - 'Variable': Feature index
                - 'Cohen d-score (w/o abs value)': Effect size with direction
    """
    n0 = len(y[y == 0])
    n1 = len(y[y == 1])

    cohen_d_matrix = []
    for i in range(np.shape(X_train)[1]):
        class0_data = X[y == 0, i]
        class1_data = X[y == 1, i]
        mu0, mu1 = np.mean(class0_data), np.mean(class1_data)
        std0, std1 = np.std(class0_data), np.std(class1_data)
        d = cohend_not_abs(mu0, mu1, std0, std1, n0, n1)
        cohen_d_matrix.append(tuple((i, d)))

    cohen_d_sorted = sorted(cohen_d_matrix, key=lambda x: x[1], reverse=True)
    df_cohen_d = pd.DataFrame(cohen_d_sorted, columns=['Variable', 'Cohen d-score (w/o abs value)'])

    return df_cohen_d


if __name__ == '__main__':
    """Main execution block that displays results and creates visualization."""
    df = separability_dataframe(X_train, y_train)
    df.to_csv('cohen_d_values.csv', index=False)
    print(df)

    feature2_class0 = X_train[y_train == 0, 2]
    feature2_class1 = X_train[y_train == 1, 2]

    plt.figure(figsize=(10, 6))
    sns.histplot(feature2_class0,
                 label='Class 0',
                 color='blue',
                 alpha=0.4,
                 kde=False,
                 stat='count')

    sns.histplot(feature2_class1,
                 label='Class 1',
                 color='orange',
                 alpha=0.4,
                 kde=False,
                 stat='count')

    plt.title('Class Histogram Comparison On Feature #2', fontsize=14, pad=20)
    plt.xlabel('Feature Values', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(fontsize=11, framealpha=1)
    plt.grid(True, alpha=0.2)

    x_min = min(feature2_class0.min(), feature2_class1.min())
    x_max = max(feature2_class0.max(), feature2_class1.max())
    amplitude = x_max - x_min
    plt.xlim(x_min - 0.05 * amplitude, x_max + 0.05 * amplitude)

    plt.tight_layout()
    plt.show()
