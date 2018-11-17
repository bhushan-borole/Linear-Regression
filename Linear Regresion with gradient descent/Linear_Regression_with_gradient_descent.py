import numpy as np 
import matplotlib.pyplot as plt 

class Linear_Regression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def generate_predicted_values(self, b):
        Y_pred = np.array([])
        for x in self.X:
            Y_pred = np.append(Y_pred, b[0] + (b[1] * x))

        return Y_pred

    def compute_cost(self, Y_pred):
        m = len(self.Y)
        J = (1 / 2*m) * (np.sum(Y_pred - self.Y)**2)
        return J

    def plot_best_fit(self, Y_pred, fig):
                f = plt.figure(fig)
                plt.scatter(self.X, self.Y, color='b')
                plt.plot(self.X, Y_pred, color='g')
                f.show()


def main():
    X = np.array([0,1,2,3,4,5,6,7,8,9,10])
    Y = np.array([0,2,4,6,8,10,12,14,16,18,20])

    regressor = Linear_Regression(X, Y)

    #intializing coefficients
    b = [0, 4]

    m = np.size(X) # number of datapoints

    iterations = 100
    learning_rate = 0.01
    costs = []
    
    #original best-fit line
    Y_pred = regressor.generate_predicted_values(b)
    regressor.plot_best_fit(Y_pred, 'Initial Best Fit Line')
    

    for _ in range(iterations):
        Y_pred = regressor.generate_predicted_values(b)
        cost = regressor.compute_cost(Y_pred)
        costs.append(cost)
        b[0] = b[0] - (learning_rate * ((1/m) * np.sum(Y_pred - Y)))
        b[1] = b[1] - (learning_rate * ((1/m) * np.sum((Y_pred - Y) * X)))

    #final best-fit line
    regressor.plot_best_fit(Y_pred, 'Final Best Fit Line')

    #plot to verify cost fuction decreases
    h = plt.figure('Verification')
    plt.plot(range(iterations), costs, color='b')
    h.show()

    print(b[0], b[1])

if __name__ == '__main__':
    main()

    




