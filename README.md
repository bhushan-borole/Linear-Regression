# Linear Regression
- This is the implementation of simmple linear regression from scratch in python.
- I have implemented with and without gradient descent.
- The equation of regression line is represented as:
	h(Xᵢ) = β₀ + β₁Xᵢ
	- h(Xᵢ) represents the predicted response value for iᵗʰ observation
	- β₀ and β₁ are the regression coeffcients and represent y-intercept and slope of regression line respectiely.

We start off by defining the values in X and Y-axis by storing them inside two variables:

```python
X = np.array([0,1,2,3,4,5,6,7,8,9,10])
Y = np.array([0,2,4,6,8,10,12,14,16,18,20])
```

### With Gradient Descent Implementation:
- Based on the initial values of parameters, best fit line looks like when plotted against the actual values of X and Y-axis:
![plot1](https://user-images.githubusercontent.com/20074475/48663282-f8433600-eab3-11e8-99ba-a3b24264db9d.png)

- Now, we will run the Gradient Descent algorithm to find the optimal values of our parameters `b[0]` and `b[1]` by running the algorithm 10 times and using learning rate's value to be 0.01:

```python
iterations = 100
learning_rate = 0.01
```

- Now, at each iteration of running Gradient Descent, we will perform the following tasks:
	- predict values for our label (Y) using our hypothesis/model: `b[0] + b[1] * x`
	- compute cost function using the actual and our predicted values of Y
	- simultaneously update the values of `b[0]` and `b[1]`
```python
    def update_coeffs(self, learning_rate):
        Y_pred = self.predict()
        Y = self.Y
        m = len(Y)
        self.b[0] = self.b[0] - (learning_rate * ((1/m) * np.sum(Y_pred - Y)))
        self.b[1] = self.b[1] - (learning_rate * ((1/m) * np.sum((Y_pred - Y) * self.X)))
```

- After running Gradient Descent 10 times and updating the values of `b[0]` and `b[1]` at each iteration, we can compare the values of our parameters and the corresponding cost functions before and after running Gradient Descent:

```python
	#Before Gradient Descent:
	b[0] = 0
	b[1] = 0
	cost = 70.0

	#After Gradient Descent:
	b[0] = -0.2709674247575545 
	b[1] = 2.0631089174216615 
	cost = 0.03538502177709652
```

- The final plot of our best fit line after finding the values of our parameters using Gradient Descent:
![plot2](https://user-images.githubusercontent.com/20074475/48663298-15780480-eab4-11e8-9bf5-ec43aed19397.png)

- Plotting the cost function wrt number of iterations(verification of our iterations):
![plot3](https://user-images.githubusercontent.com/20074475/48663304-24f74d80-eab4-11e8-94d3-6f0119868b83.png)


### Without Gradient Descent Implementation:

- To create our model, we must “learn” or estimate the values of regression coefficients `b[0]` and `b[1]`. And once we’ve estimated these coefficients, we can use the model to predict responses!

- In this article, we are going to use the Least Squares technique.
- In this method we directly calculate the estimated regression coefficients from the given datasets.
```python
	def get_estimate_coeff(x, y):
		n = np.size(x)

		mx, my = np.mean(x), np.mean(y)
		SSxy = np.sum((y * x) - (n * mx * my))
		SSxx = np.sum((x * x) - (n * mx * mx))

		b1 = SSxy / SSxx
		b0 = my - b1 * mx

		return [b0, b1]

	'''
	Estimated coeffcients: 
	b0: 0.0 
	b1: 2.0
	'''
```
- After finding the estimated coefficients we predict the value of y and plot the best-fit line.
![plot1](https://user-images.githubusercontent.com/37565807/48694486-6bf35900-ec02-11e8-8244-4b9b8c612c86.png)





