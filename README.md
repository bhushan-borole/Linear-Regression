# Regression
- This is the implementation of simmple linear regression from scratch in python.
- I have implemented with and without gradient descent.
- The equation of regression line is represented as:
	h(Xᵢ) = β₀ + β₁Xᵢ
	- h(Xᵢ) represents the predicted response value for iᵗʰ observation
	- β₀ and β₁ are the regression coeffcients and represent y-intercept and slope of regression line respectiely.

We start off by defining the values in X and Y-axis by storing them inside two variables:

```
X = np.array([0,1,2,3,4,5,6,7,8,9,10])
Y = np.array([0,2,4,6,8,10,12,14,16,18,20])
```

- With Gradient Descent Implementation:
Based on the initial values of parameters, this is how our best fit line looks like when plotted against the actual values of X and Y-axis:
![alt text](https://github.com/bhushan-borole/Linear-Regression/blob/master/Linear%20Regresion%20with%20gradient%20descent/plot1.png)

Now, we will run the Gradient Descent algorithm to find the optimal values of our parameters `theta_0` and `theta_1` by running the algorithm 10 times and using learning rate's value to be 0.01:

```
iterations = 10
learning_rate = 0.01
```

Now, at each iteration of running Gradient Descent, we will perform the following tasks:
- predict values for our label (Y) using our hypothesis/model: `b[0] + b[1] * x`
- compute cost function using the actual and our predicted values of Y
- simultaneously update the values of `b[0]` and `b[1]`


- Without Gradient Descent Implementation:
	- To create our model, we must “learn” or estimate the values of regression coefficients b_0 and b_1. And once we’ve estimated these coefficients, we can use the model to predict responses!

	- In this article, we are going to use the Least Squares technique.
	- β₁ = SSxy / SSxx
	- β₀ = y_mean - β₁ * x_mean




