import numpy as np


def cross_validation_scores(model, X, y, folds, metric):
    """
    Perform cross-validation for a given model and data
    :param model: a model object with fit and predict methods
    :param X: np.array of shape (n_samples, n_features)
    :param y: np.array of shape (n_samples,)
    :param folds: a k-folds object with split method
    :param metric: evaluation metric function based on y_true and y_pred
    :return: metric scores as list
    """
    scores = []  # list of results per fold
    for train_index, val_index in folds.split(X):
        # obtain the split of the entire training data into a training-set and a validation-set for this folds-split
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        scores.append(metric(y_val, y_val_pred))

    return scores
