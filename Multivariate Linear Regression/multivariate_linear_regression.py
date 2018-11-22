import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#read data
data = pd.read_csv('home.txt',names=["size","bedroom","price"])

#normalizing data
'''
Normalization is a technique often applied as part of data preparation for machine learning. 
The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, 
without distorting differences in the ranges of values or losing information
'''
data = (data - data.mean()) / data.std()

X = data.iloc[:, 0:2]

'''
numpy.ones(shape, dtype, order)
@param shape : integer or sequence of integers
@param order : C_contiguous or F_contiguous
			   C-contiguous order in memory(last index varies the fastest)
			   C order means that operating row-rise on the array will be slightly quicker
			   FORTRAN-contiguous order in memory (first index varies the fastest).
			   F order means that column-wise operations will be faster.
@param dtype : [optional, float(byDeafult)] Data type of returned array.  
'''
ones = np.ones([X.shape[0], 1])
#print(ones)
X = np.concatenate((ones, X), axis = 1)
#print(X)

Y = data.iloc[:, 2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1, 3])

learning_rate = 0.01
iterations = 1000

#compute cost
def compute_cost(X, Y, theta):
	sum = np.power(((X @ theta.T) - Y), 2)
	return np.sum(sum / (2 * len(X)))


#gradient descent
'''
X @ theta.T is a matrix operation.
It does not matter how many columns are there in X or theta, 
as long as theta and X have the same number of columns the code will work.

We could have used for loops to do the same thing, 
but why use inefficient `for loops` when we have access to NumPy.
'''
def gradient_descent(X, Y, theta, iterations, learning_rate):
	cost = np.zeros(iterations)
	for i in range(iterations):
		theta -= (learning_rate/len(X)) * np.sum(X * (X @ theta.T - Y), axis = 0)
		cost[i] = compute_cost(X, Y, theta)

	return theta, cost

gradient, cost = gradient_descent(X, Y, theta, iterations, learning_rate)
print(gradient, cost)

final_cost = compute_cost(X, Y, gradient)
print(final_cost)

fig, ax = plt.subplots()
ax.plot(np.arange(iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error Vs. Training Epochs')
'''
model = linear_model.LinearRegression()
model.fit(X, Y)
x = np.ravel(X[:, 1])
f = model.predict(X).flatten()

fig, ax = plt.subplots()
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['size'], data['price'])
ax.legend(loc=2)
ax.set_xlabel('Price')
ax.set_ylabel('Size')
ax.set_title('Predicted Size vs. Price')
'''
plt.show()






