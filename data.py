import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
np.random.seed(42)


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.0001^2)
    """
    noise = np.random.normal(loc=0, scale=0.0001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=42)

def load_data(path):
    """ reads and returns the pandas DataFrame """
    df = pd.read_csv(path)
    return df

def adjust_labels(y):
    """adjust labels of season from {0,1,2,3} to {0,1}"""
    x = y.copy()
    x[x == 1] = 0
    x[x == 2] = 1
    x[x == 3] = 1
    return x


class StandardScaler:
    def __init__(self):
        """object instantiation
        Attributes
        ----------
        _mean: mean of train data
        _std: std of train data"""
        self._mean = None
        self._std = None

    def fit(self, X):
        """ fit scaler by learning the mean and standard deviation per feature """
        self._mean = np.mean(X, axis = 0)
        self._std = np.std(X, axis = 0, ddof = 1)


    def transform(self, X):
        """ transform X by learned mean and standard deviation, and return it """
        return (X - self._mean) / self._std
    
    def fit_transform(self, X):
        """ fit scaler by learning the mean and standard deviation per feature, and then transform X """
        self.fit(X)
        return self.transform(X)


    

    

