import sys
import pandas as pd
import numpy as np
import data
from cross_validation import cross_validation_scores
import knn 
import evaluation


def PartA(df, folds, k_list):
    """
    Classification part of the algorithm.

    :param df: Input DataFrame containing the dataset.
    :param folds: Number of folds for cross-validation.
    :param k_list: List of k values to evaluate the algorithm’s performance on.
    :return: None
    """
    X = df[['t1', 't2', 'wind_speed','hum']].to_numpy()
    y = data.adjust_labels(df.season)
    X = data.add_noise(X) 
    graph = []
    for k in k_list:
        scores = cross_validation_scores(knn.ClassificationKNN(k), X, y, folds, evaluation.f1_score)
        mean = np.mean(scores)
        std = np.std(scores, ddof = 1)
        graph.append(mean)
        print(f'k={k}, mean score: {round(mean,4):.4f}, std of scores: {round(std,4):.4f}')
    evaluation.visualize_results(k_list, graph, evaluation.f1_score, 'Classification', 'plot1')

    

def PartB(df, folds, k_list):
    """
    Regression part of the algorithm.

    :param df: Input DataFrame containing the dataset.
    :param folds: Number of folds for cross-validation.
    :param k_list: List of k values to evaluate the algorithm’s performance on.
    :return: None
    """
    X = df[['t1', 't2', 'wind_speed']].to_numpy()
    y = df['hum'].to_numpy()
    X = data.add_noise(X) 
    graph = []
    for k in k_list:
        scores = cross_validation_scores(knn.RegressionKNN(k), X, y, folds, evaluation.rmse)
        mean = np.mean(scores)
        std = np.std(scores, ddof = 1)
        graph.append(mean)
        print(f'k={k}, mean score: {round(mean,4):.4f}, std of scores: {round(std,4):.4f}')
    evaluation.visualize_results(k_list, graph, evaluation.rmse, 'Regression', 'plot2')


def main():
    df = data.load_data('london_sample_2500.csv')
    folds = data.get_folds()
    k_list = [3,5,11,25,51,75,101]

    print('Part1 - Classification')
    PartA(df, folds, k_list)
    print()
    print('Part2 - Regression')
    PartB(df, folds, k_list)
    print()
    return 0
  
if __name__ == '__main__':
    sys.exit(main())

