"""
This is an example of how to submit the coursework. This file MUST contain:

  - fit_preprocess(data_path): A function that takes a single parameter, the 
  path to the CSV file containing the data. It returns an object (e.g., a 
  dictionary) with parameters used to preprocess the data.
  - load_and_preprocess(data_path, preprocess_params): A function with two 
  parameters: the path to the CSV file containing the data and the 
  preprocessing parameters object returned by fit_preprocess. It returns two 
  objects: the preprocessed data and the labels, which can be in the form of 
  numpy arrays, pandas DataFrames, or Python lists.
  - fit_model(X): A function that takes the preprocessed data as a single 
  parameter and returns a trained model, which will be used by predict. 
  This model is any object that predict requires to perform the detection task.
  - predict(X, model): A function that returns the anomaly detection prediction 
  for the input data, X,  based on the provided model.

Do not change the name of "fit_preprocess", "load_and_preprocess", "fit_model"
and "predict" functions.

Separating the code into several functions is based on the idea of running
the training of the model only once, and make predictions several times with
the already trained model.

You can include any other code that is useful for the coursework. For instance,
here `main()` is included for us to check that predict and model run properly,
and to show some results of the performance of the model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def fit_preprocess(data_path):
    """
    Return a single object representing the parameters used to preprocess the
    data.
    In this example, we return a dictionary with the mean and standard
    deviation for standardization.

    Parameters
    ----------
    data_path : str
        Path to the csv file containing the data.

    Returns
    -------
    preprocess_params : dict
        Dictionary containing the parameters used to preprocess the data.
    """
    # Load from file
    data = pd.read_csv(data_path)
    # Obtain preprocess parameters
    X = data['XMEAS(1)'].to_numpy()
    mean = X.mean()
    std = X.std()
    preprocess_params = {'mean': mean, 'std': std}
    return preprocess_params


def load_and_preprocess(data_path, preprocess_params):
    """
    This function loads and preprocesses data from a csv file
    In this example, we read the first feature as the variable of interest,
    then standardize the data.

    Parameters
    ----------
    data_path : str
        Path to the csv file containing the data.
    preprocess_params : dict
        Dictionary containing the parameters used to preprocess the data.

    Returns
    -------
    X : 2D numpy array
        Preprocessed data. Make sure X is of shape: (n_instances, n_features).
    y : 1D numpy array
        True normal/anomaly labels of the data.
    """
    # Load from file
    data = pd.read_csv(data_path)
    # Separate label and choose features, then reshape to 2D array
    X = data[['XMEAS(1)']].to_numpy()
    y = data['label'].to_numpy()
    # Preprocess data
    mean = preprocess_params['mean']
    std = preprocess_params['std']
    X = (X - mean) / std
    return X, y


def fit_model(X):
    """
    Returns the fitted model. In this simple example, the anomaly detection
    model is the threshold that will flag anomalies, but it can be any object
    as long as the predict() function knows how to handle it when passed as an
    argument.

    Parameters
    ----------
    X : 2D numpy array.
        Already preprocessed data used to fit the model.

    Returns
    -------
    model : object
        Fitted model.
    """
    # Fit distribution
    mean = X.mean()
    std = X.std()
    
    # Determine threshold
    threshold = mean + std
    return threshold
    

def predict(X, model):
    """
    Accepts a 2D numpy array and an anomaly detection model, and returns a 1D
    array of predictions.

    Parameters
    ----------
    X : 2D numpy array.
        Preprocessed data.
    model : object
        Anomaly detection model.

    Returns
    -------
    y_pred : 1D numpy array.
        Array of normal/anomaly predictions.
    """
    n = len(X)
    y_pred = np.zeros((n,))
    for i in range(len(X)):
        if X[i] > model:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


def main():
    """
    Main function to test the code when running this script standalone. Not
    mandatory for the deliverable but useful for the grader to check the code
    workflow and for the student to check it works as expected.
    """
    train_path = 'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_train.csv'
    test_path = 'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_test.csv'

    # Set up preprocessing
    preprocess_params = fit_preprocess(train_path)

    # Load and preprocess data
    X_train, y_train = load_and_preprocess(train_path, preprocess_params)
    X_test, y_test = load_and_preprocess(test_path, preprocess_params)

    # Get the detection model
    threshold = fit_model(X_train)

    # Check performance on data
    # (just for us to see it works properly, the grader might use other data)
    y_pred_train = predict(X_train, threshold)
    print('Training accuracy: ', accuracy_score(y_train, y_pred_train))
    y_pred_test = predict(X_test, threshold)
    print('Test accuracy: ', accuracy_score(y_test, y_pred_test))


if __name__ == '__main__':
    main()
    