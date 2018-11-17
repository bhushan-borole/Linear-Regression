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

We have also initialized our parameter value `b[1]` to initially be equal to 4:

```
# initializing theta parameters
b = [0, 4]
```


