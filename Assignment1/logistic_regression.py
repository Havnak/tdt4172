import numpy as np
import  pandas as pd


class LogisticRegression():
    
    def __init__(self, learning_rate=0.00001, epochs = 10000000, presicion = 1e-15):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.presicion = presicion
        self.weights = None
        
    def cost(self, X, y):
        return -(1/self.m) * np.sum(y*np.log(self.a)+(1-y)*np.log(1-self.a))

    def db(self, y):
        return (1/self.m) * np.sum(self.a-y)

    def dw(self, X, y):
        return (1/self.m) * np.dot(self.a-y, X.T)

    def update_weights(self, X, y):
        prev_w = self.w.copy()

        self.b -= self.db(y) * self.learning_rate
        self.w -= self.dw(X,y).T * self.learning_rate
        return prev_w

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        self.m = X.shape[1]
        n = X.shape[0]
        self.weights = np.ones((n+1,1))-0.5
        self.b = self.weights[0]
        self.w = self.weights[1:]
        diff =[np.inf] * len(self.weights) 
        steps = 0
        cost_list = []
        

        for steps in range(self.epochs):
            self.z = np.dot(self.w.T, X)+self.b
            self.a = 1/(1+np.exp(-self.z))
            prev = self.update_weights(X,y)
            diff = prev - self.w
            steps += 1

            cost_list.append(self.cost(X,y))

            if not all([abs(difference)>self.presicion for difference in diff]):
                break
        return cost_list
    
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
        self.z = np.matmul(self.w.T, X)+self.b
        a = 1/(1+np.exp(-self.z))
        return np.array(a>0.5, dtype='int64')

    def accuracy(self, X, y):
        accuracy = np.average(1-np.abs(self.predict(X)-y))
        print(f'The accuracy is {accuracy}')
        return accuracy

        
        

