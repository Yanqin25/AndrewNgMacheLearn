import numpy as np
import matplotlib.pyplot as plt
import math


def displayData(X, example_width=None):
    '''Display 2D data in a nice grid
    [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    stored in X in a nice grid. It returns the figure handle h and the 
    displayed array if requested.'''
    plt.figure()
# Set example_width automatically if not passed in
    (m, n) = X.shape  # 100*400
    example_width = example_width if example_width else math.floor(np.sqrt(n))
    example_height = math.ceil(n/example_width)

# Compute number of items to display
    display_rows = math.floor(np.sqrt(m))
    display_cols = math.ceil(m/display_rows)
# Between images padding
    pad = 1
# Setup blank display
    display_array = np.ones(
        (pad+display_rows*(example_height+pad), pad+display_cols*(example_width+pad)))
    curr_cell = 0
    # for i in range(display_rows):
    #     for j in range(display_cols):
    #         curr_cell=i*display_cols+j
    #         max_val = np.max(X[curr_cell, :])
    #         row = pad+i*(example_height+pad)+np.arange(example_height)
    #         col = pad+j*(example_width+pad)+np.arange(example_width)
    #         display_array[np.min(row):np.max(row)+1, np.min(col):np.max(col) +
    #                       1] = X[curr_cell, :].reshape(example_height, example_width).T

    for i in range(display_rows):
        for j in range(display_cols):
            curr_idx=i*display_cols+j
            curr_cell=X[curr_idx, :].reshape(example_height, example_width)
            display_array=np.pad(curr_cell,pad_width=1,mode='constant',constant_values=1)
            old_col=np.c_[old_col,display_array]
        old_row=np.r_[old_row,old_col]
    plt.imshow(display_array, cmap=plt.get_cmap('gray_r'))

