import numpy as np
import  pandas as pd

class LinearRegression():
    
    def __init__(self, learning_rate=0.000001, step_limit = 100000, presicion = 0.00000001):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.step_limit = step_limit
        self.presicion = presicion
        self.b = None

    def lstsq_intersect(self, X, y):
        return -2*np.sum(y-self.b[0]-np.dot(X,self.b[1:]))

    def lstsq(self, X, y):
        return -2 * np.sum(np.diag(np.dot(X.T, (y - self.b[0] - np.dot(X, self.b[1:])))))

    def update_value(self, X, y):
        prev_b = self.b.copy()

        self.b[0] -= self.lstsq_intersect(X, y) * self.learning_rate
        self.b[1:] -= self.lstsq(X,y) * self.learning_rate
        return prev_b

    def output_weights(self):
        return list(self.b)

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        X = X.values
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.b = np.zeros(X.shape[1] + 1)
        diff =[np.inf] * len(self.b) 
        steps = 0

        while all([abs(difference)>self.presicion for difference in diff]) and steps<self.step_limit:
            prev = self.update_value(X,y)
            diff = prev - self.b
            steps += 1
        
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        return self.b[0] + X*self.b[1:]