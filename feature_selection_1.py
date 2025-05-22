from data_loader import X_train, X_test, y_train, y_test
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

rs = 42

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=rs)),
    ('clf', LogisticRegression(random_state=rs, max_iter=1000))  # Increased max_iter for convergence
])

# Expanded parameter grid to tune both PCA components AND logistic regression's C
param_grid = {
    'pca__n_components': list(range(2, 31)),  # Test from 2 to 30 components
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strengths (smaller C = stronger regularization)
    'clf__penalty': ['l2']
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Parallelize computation
)

# Fit the grid search to find the best parameters
grid_search.fit(X_train, y_train)

# Get the best parameters
best_n_components = grid_search.best_params_['pca__n_components']
best_C = grid_search.best_params_['clf__C']
print(f"\nBest number of PCA components: {best_n_components}")
print(f"Best regularization strength (C): {best_C}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Evaluate on the test set using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy with best model: {(100 * test_accuracy):.1f}%")

# Plot CV results
cv_results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(12, 6))

# Plot accuracy vs n_components for best C value
best_C_results = cv_results[cv_results['param_clf__C'] == best_C]
plt.plot(best_C_results['param_pca__n_components'],
         best_C_results['mean_test_score'],
         label=f'C={best_C}', linewidth=2)

plt.xlabel('Number of PCA Components')
plt.ylabel('CV Accuracy')
plt.title(f'Accuracy vs. PCA Components (Best C={best_C})')
plt.legend()
plt.grid()
plt.show()
