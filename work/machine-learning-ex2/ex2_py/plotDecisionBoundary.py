import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from mapFeature import mapFeature


def plotDecisionBoundary(theta, X, y):
    '''Plots the data points X and y into a new figure with the decision boundary defined by theta'''
#   plots the data points with + for the
#   positive examples and o for the negative examples. X is assumed to be
#   a either
#   1) Mx3 matrix, where the first column is an all-ones column for the
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones

# Plot Data
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1])-2, np.max(X[:, 1])+2])

        # Calculate the decision boundary line
        # plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1))
        plot_y = -(plot_x*theta[1]+theta[0])/theta[2]

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y, label='Decision Boundary')

        # Legend, specific for the exercise
        plt.axis([30, 100, 30, 100])

    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        # for i = 1:length(u)
        #     for j = 1:length(v)
        #         z(i,j) = mapFeature(u(i), v(j))*theta
        #     end
        # end

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(np.array(u[i]), np.array(v[j])).dot(theta)
        u, v = np.meshgrid(u, v)
        z = z.T  # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, levels=0)
