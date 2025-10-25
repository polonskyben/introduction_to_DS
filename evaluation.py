import numpy as np
import matplotlib.pyplot as plt

def f1_score(y_true, y_pred):
    """returns f1_score of binary classification task with true labels y_true and predicted labels y_pred """
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FP) != 0 else 0
    return 2 * precision * recall / (precision + recall)    

def rmse (y_true, y_pred):
    """returns RMSE of regression task with true labels y_true and predicted labels y_pred"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def visualize_results(k_list, scores, metric, title, path):
    """ plot a results graph of cross validation scores """
    plt.plot(k_list, scores)
    plt.title(title)
    plt.ylabel(metric.__name__)
    plt.xlabel('k')
    plt.savefig(path)
    plt.close()
