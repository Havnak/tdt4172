import numpy as np
from linear_regression import LinearRegression 

lr = LinearRegression()

X = np.matrix('''
1;
2;
3
''')
y = np.array([2,4,6])
b = [0,0.5]

print(lr.lstsq_intersect(X, y, b))

print(lr.lstsq(X,y,b))
