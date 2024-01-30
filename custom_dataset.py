from math import floor
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import numpy as np
class customDataset:
    """
    Load UCI dataset.

    Args:
        dataset: name of the dataset to load. This can be either the name of the directory
            that the dataset is in OR the identifier used in papers. For example you can
            specify dataset='houseelectric' OR dataset='electric' and it will give you the
            same thing. This allows for convienent abbreviations.
        print_stats: if true then will print stats about the dataset.
    """

    def __init__(self, dataset: str, dtype=np.float64, print_stats: bool = True,section = 1):
        assert isinstance(dataset, str), "dataset must be a string"
        dataset = dataset.lower()  # convert to lowercase
        dataset = dataset.replace(" ", "")  # remove whitespace
        dataset = dataset.replace("_", "")  # remove underscores

        
        try:
            if dataset == "california":
                X , Y = fetch_california_housing(return_X_y=True)
                # take only 50% of the data because it is too big
                # shuffle the data
                np.random.seed(42)
                idx = np.random.permutation(len(X))
                X = X[idx]
                Y = Y[idx]
                self.X = X[:int(len(X)/section)]
                self.Y = Y[:int(len(Y)/section)]
            elif dataset == "boston":
                

                data_url = "http://lib.stat.cmu.edu/datasets/boston"
                raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
                self.X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
                self.Y = raw_df.values[1::2, 2]
        except:
            print("Load failed, maybe dataset string is not correct.")
            raise

        

    def get_split(
        self, split: int = 0
    ) :
        """
        Split the data in 10 folds
        """
        X = self.X
        y = self.Y
        i = split
        n_observation = X.shape[0]
        n_observation_per_fold = floor(n_observation/10)
        X_folds_test = X[i*n_observation_per_fold:(i+1)*n_observation_per_fold]
        y_folds_test = y[i*n_observation_per_fold:(i+1)*n_observation_per_fold]
        X_folds_train = np.concatenate((X[:i*n_observation_per_fold], X[(i+1)*n_observation_per_fold:]))
        y_folds_train = np.concatenate((y[:i*n_observation_per_fold], y[(i+1)*n_observation_per_fold:]))

     
        


        return X_folds_train, y_folds_train, X_folds_test, y_folds_test


