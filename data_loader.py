import numpy as np
import os
from sklearn.model_selection import train_test_split


def convert(text_data, as_float=True):
    """Converts a space/newline-separated string into a NumPy array.

    Args:
        text_data (str): Input string with space-separated values and newline-separated rows.
        as_float (bool, optional): If True, converts values to float; if False, converts to int. Defaults to True.

    Returns:
        np.ndarray: A 2D array if `as_float=True` (for X matrix), or a 1D array if `as_float=False` (for 0-1 target y).
    """
    text_data = text_data.strip()
    text_data = text_data.split('\n')
    if as_float:
        matrix = [list(map(float, line.split())) for line in text_data]
        matrix = np.array(matrix)
    else:
        matrix = [list(map(int, line.split())) for line in text_data]
        matrix = np.array(matrix)
        matrix.reshape(-1)
    return matrix


def export(matrix, as_float=True):
    """Converts a NumPy array into a space/newline-separated string.

    Args:
        matrix (np.ndarray): Input array (1D or 2D).
        as_float (bool, optional): If False, casts values to int before conversion. Defaults to True.

    Returns:
        str: Space-separated values with rows newline-separated.
    """
    if not as_float:
        matrix = matrix.astype(int)
    text_data = [' '.join(map(str, row)) for row in matrix]
    text_data = '\n'.join(text_data)
    return text_data


def get_data(file_name):
    """Loads and converts data from a file in the 'data' directory.

    Args:
        file_name (str): Name of the file (e.g., 'x_train.txt'). Files starting with 'x' are parsed as floats,
                         and files starting with 'y' are parsed as integers.

    Returns:
        np.ndarray: Converted data array.
    """
    file_path = os.path.join("data", file_name)
    with open(file_path, "r") as f:
        text = f.read()
    if file_name[0] == 'x':
        data = convert(text, as_float=True)
    elif file_name[0] == 'y':
        data = convert(text, as_float=False)
    else:
        raise ValueError(f"Invalid file name: '{file_name}'. Must start with 'x' or 'y'.")
    return data


X, X_prediction, y = get_data('x_train.txt'), get_data('x_test.txt'), get_data('y_train.txt')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
y_train, y_test = y_train.ravel(), y_test.ravel()

if __name__ == '__main__':
    print(f'Training data shape: {X_train.shape}.')
    print(f'Training target variable shape: {y_train.shape}.')
    print(f'Testing data shape: {X_test.shape}.')
    print(f'Testing target variable shape: {y_test.shape}.')
    print(f'Predictions data shape: {X_prediction.shape}.')
