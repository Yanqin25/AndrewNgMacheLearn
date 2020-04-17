import numpy as np
import matplotlib.pyplot as plt 
def plotData(X, y):
    '''plots the data points with + for the positive examples and o for the negative examples. X is assumed to be a Mx2 matrix.'''

# Create New Figure

    plt.figure()
# ====================== YOUR CODE HERE ======================
# Instructions: Plot the positive and negative examples on a
#               2D plot, using the option 'k+' for the positive
#               examples and 'ko' for the negative examples.
#
# markerfacecolor
    X1=X[y==1,:]
    # X1=X[np.where(y==1,True,False).flatten()]
    X0=X[y==0,:]
    plt.plot(X1[:,0],X1[:,1],'+k',linewidth=2,markersize=7)
    plt.plot(X0[:,0],X0[:,1],'ok',markerfacecolor='yellow',markersize=7)
    plt.title('Scatter plot of trainning data')
# =========================================================================

