import numpy as np


class KNN:
    
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=1):
       
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_one_loop(self, X):
      
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        
        for i_test in range(num_test):
        
            dists[i_test] = np.sum(np.abs(X[i_test] - self.train_X), axis=1)
            
        return dists

    def predict_labels_binary(self, dists):

        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
    
            unique, counts = np.unique(self.train_y[dists[i].argsort()[:self.k]], return_counts = True)
            diction = dict(zip(unique, counts))
            maximum = sorted(diction.items(), key= lambda x: x[1])[-1][0]
            pred[i] = (maximum == 1)

        return pred

