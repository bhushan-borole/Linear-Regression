import numpy as np 
import matplotlib.pyplot as plt 

class Linear_Regression:
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def generate_predicted_values(self, X, b):
		Y_pred = np.array([])
		for x in X:
			Y_pred = np.append(Y_pred, b[0] + (b[1] * x))

		return Y_pred

	def compute_cost(self, Y, Y_pred):
		m = len(Y)
		J = (1 / 2*m) * (np.sum(Y_pred - Y)**2)
		return J

	def get_estimate_coeff(self, x, y):
		n = np.size(x)

		mx, my = np.mean(x), np.mean(y)
		SSxy = np.sum((y * x) - (n * mx * my))
		SSxx = np.sum((x * x) - (n * mx * mx))

		b1 = SSxy / SSxx
		b0 = my - b1 * mx

		return [b0, b1]

	def plot(self, iterations, costs):
		plt.plot(range(iterations), costs)
		plt.show()


def main():
	X = np.array([0,1,2,3,4,5,6,7,8,9,10])
	Y = np.array([0,2,4,6,8,10,12,14,16,18,20])

	regressor = Linear_Regression(X, Y)

	#getting estimated coefficients
	#b = regressor.get_estimate_coeff(X, Y)
	b = [0, 4]

	m = np.size(X) # number of datapoints

	iterations = 100
	learning_rate = 0.001
	costs = []

	for _ in range(iterations):
		Y_pred = regressor.generate_predicted_values(X, b)
		cost = regressor.compute_cost(Y, Y_pred)
		costs.append(cost)
		b[0] = b[0] - (learning_rate * ((1/m) * np.sum(Y_pred - Y)))
		b[1] = b[1] - (learning_rate * ((1/m) * np.sum((Y_pred - Y) * X)))
	regressor.plot(iterations, costs)

if __name__ == '__main__':
	main()

	




