import numpy as np
import  pandas as pd

class LogisticRegression():
    
    def __init__(self, learning_rate=0.000001, step_limit = 100000, presicion = 0.00000001):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.step_limit = step_limit
        self.presicion = presicion
        self.w = None
        

    def cost_constant(self, X, y):
        return (self.a-y)

    def cost(self, X, y):
        return (self.a-y)*X

    def update_value(self, X, y):
        prev_w = self.w.copy()

        self.w[0] -= self.cost_constant(X, y) * self.learning_rate
        self.w[1:] -= self.cost(X,y) * self.learning_rate
        return prev_w

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        X = X.values
        y = y.values

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.w = np.zeros(X.shape[1] + 1)
        diff =[np.inf] * len(self.w) 
        steps = 0

        b = self.w[0].copy()
        wt = self.w[1:].T.copy()
        self.z = wt*X+b
        self.a = 1/(1+np.exp(-self.z))

        while all([abs(difference)>self.presicion for difference in diff]) and steps<self.step_limit:
            prev = self.update_value(X,y)
            diff = prev - self.w
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
        b = self.w[0].copy()
        wt = self.w[1:].T.copy()
        z = wt*X+b
        return 1/(1+np.exp(-z))

