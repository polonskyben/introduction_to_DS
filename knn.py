import numpy as np
from statistics import mode
from abc import abstractmethod, ABC
from data import StandardScaler


class KNN(ABC):
    def __init__(self, k):
        """object instantiation, save k and define a scaler object 
        Parameters
        ----------
        k: number of nearest neighbours to use.

        Attributes
        ----------
        k: number of neighbours.
        scaler: scaling model.
        X_train: scaled train data.
        y_train: labels of the train data.
        """
        self.k = k
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """ fit scaler and save X_train and y_train
        :X_train: scaled train data
        :y_train: numpy array of labels of the train data"""
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = np.array(y_train)

    @abstractmethod
    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        pass

    def neighbours_indices(self, x):
        """ for a given point x, find indices of k closest points in the training set
        :x: point"""
        #we skip np.sqrt because we need only relative distance (ordering is preserved)
        dist_v = np.sum((self.X_train - x) ** 2, axis=1)
        # Reference (source: stackoverflow): A fast way to find N largest elemnts in an numpy array
        # https://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array
        return np.argpartition(dist_v, self.k)[:self.k]
        
    @staticmethod
    def dist(x1, x2):
        """returns Euclidean distance between x1 and x2"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    

class RegressionKNN(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation """
        super().__init__(k)
    
    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        X_test = self.scaler.transform(X_test)
        predicted_labels = np.zeros(len(X_test))
        for i,x in enumerate(X_test):
            list_k = self.neighbours_indices(x)
            predicted_labels[i] = np.mean(self.y_train[list_k])
        return predicted_labels


class ClassificationKNN(KNN):
    def __init__(self, k):
        """ object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """
        X_test = self.scaler.transform(X_test)
        predicted_labels = np.zeros(len(X_test), dtype = int)
        for i,x in enumerate(X_test):
            list_k = self.neighbours_indices(x)
            predicted_labels[i] = mode(self.y_train[list_k])
        return predicted_labels
    
    