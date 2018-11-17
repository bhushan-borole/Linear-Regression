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

- Without Gradient Descent Implementation:
	- To create our model, we must “learn” or estimate the values of regression coefficients b_0 and b_1. And once we’ve estimated these coefficients, we can use the model to predict responses!

	- In this article, we are going to use the Least Squares technique.

	- Now consider:
		- yᵢ = β₀ + β₁Xᵢ + εᵢ = h(Xᵢ) + εᵢ => εᵢ = yᵢ - h(Xᵢ)
		- here εᵢ is residual error in iᵗʰ observation.
		- So our aim is to minimize the total residual error.
		- We define the squared error or cost function, J as:
			![formula](/readmes/img/7e461e493791e490950b5706050ee9242220dfac.latex "J(β₀ + β₁) = \frac{1}{2n}\sum_{i=1}^nεᵢ²")
			"renderer": "http://chart.googleapis.com/chart?cht=tx&chl={J(β₀ + β₁) = \frac{1}{2n}\sum_{i=1}^nεᵢ²}"


