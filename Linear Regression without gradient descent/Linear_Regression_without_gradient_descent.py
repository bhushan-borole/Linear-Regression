import numpy as np
import matplotlib.pyplot as plt

def get_estimate_coeff(x, y):
	n = np.size(x)

	mx, my = np.mean(x), np.mean(y)
	SSxy = np.sum((y * x) - (n * mx * my))
	SSxx = np.sum((x * x) - (n * mx * mx))

	b1 = SSxy / SSxx
	b0 = my - b1 * mx

	return [b0, b1]

def plot(x, y, b):
	plt.scatter(x, y, color = 'r')

	h = b[0] + b[1] * x
	plt.plot(x, h, color = 'g')
	plt.xlabel('X')
	plt.ylabel('Y')

	plt.show()



def main():
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	y = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

	b = get_estimate_coeff(x, y)
	print("Estimated coeffcients: \nb0: {} \nb1: {} ".format(b[0], b[1]))
	plot(x, y, b)

if __name__ == '__main__':
	main()


