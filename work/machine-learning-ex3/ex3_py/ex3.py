from predictOneVsAll import predictOneVsAll
from oneVsAll import oneVsAll
from lrCostAndGradFunction import lrCostFunction, lrGradFunction
from displayData import displayData
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Initialization

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

# training data stored in arrays X, y
data = loadmat('ex3data1.mat')
X = data['X']
# ravel is very important here,y must be row vector!
y = data['y'].ravel()
m = X.shape[0]


# Randomly select 100 data points to display
# rand_indices = randperm(m)
# rand_indices=random.sample(range(m),m)
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[:100], :]
sel_y=y[rand_indices[:100]]
displayData(sel)

print('Program paused. Press enter to continue.\n')


# ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

# # Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape(3, 5).T/10]
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3
J = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = lrGradFunction(theta_t, X_t, y_t, lambda_t)
print('\nCost: #f\n', J)
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(' #f \n', grad)
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

input('Program paused. Press enter to continue.\n')

# ## ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

lambda2 = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda2)

print('Program paused. Press enter to continue.\n')


# ## ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta.T, X)
print('\nTraining Set Accuracy: %f\n' % (np.mean(pred == y) * 100))

## predict for train data(random 100 rows):
# pred2=predictOneVsAll(all_theta.T,sel)
# print(pred2.reshape(10,10))
# print('random 100 rows accuracy:{}'.format(np.mean(pred2==sel_y)*100))
plt.show()
