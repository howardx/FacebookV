import pandas as pd
import numpy as np

from itertools import tee, izip
import os

# a helper function takes an iterable, return its stepwise pair tuple in a list
# [1, 2, 3, 4, 5] -> [(1, 2), (2, 3), (3, 4), (4, 5)]
def pairwise(iterable):
    floor, ceiling = tee(iterable)
    next(ceiling, None)
    return izip(floor, ceiling)

# a helper function takes a df, a column name, a list of floor/ceiling values
# "split" the df based on the given column, using the list of floor/ceiling values, return list of df
#
# flr_clg will be floor EXclusive, ceiling INclusive - need special treatment for the first split
def split_df_rows_on_col_ranges(df, col, flr_clg):
    splitted_df = []
    first = True
    for fc in flr_clg:
        if first:
            splitted_df.append(df[ (df[col] >= fc[0]) & (df[col] <= fc[1]) ])
            first = False
        else:
            splitted_df.append(df[ (df[col] > fc[0]) & (df[col] <= fc[1]) ])
    return splitted_df

# a helper function takes x bars, cut y bars inside and return a dictionary of grids
def cut_y_bars_in_x_bar(x_bars, y, y_bin_tuple):
    gridDict = {}
    xidx = 0
    for xbar in x_bars:
        # getting list of N bars (grids here already) based on y values, all within 1 xbar
        y_bars_in_xbar = split_df_rows_on_col_ranges(xbar, y, y_bin_tuple)

        yidx = 0
        for grid in y_bars_in_xbar:
            gridDict[(xidx, yidx)] = grid # gather output with x,y index
            yidx = yidx + 1
        xidx = xidx + 1
    return gridDict

'''
input parameters for def get_grids():
- train - input filename/dataframe for training set

- test - input filename/dataframe for test set

- outputFile - boolean that tells whether you want NxM files as output or a dict of pd.DataFrame
               as output, format would be (x_idx, y_idx) : df_for_grid. If you want file as output
               then x_idx, y_idx will appear in output files' name

- train_output - only used if the 3rd parameter is set to True, will be the path to store NxM files for training set,
                 each file contains a grid of data points

- test_output - only used if the 3rd parameter is set to True, will be the path to store NxM files for testing set,
                each file contains a grid of data points

- n - NxM grid, the N value, for x axis

- m - NxM grid, the M value, for y axis

- x - column name of the x coordinate in input file

- y - column name of the y coordinate in input file

'''
def get_grids(train, test, outputFile = False, train_output = None, test_output = None, n = 10, m = 10, x = 'x', y = 'y'):
    if isinstance(train, basestring):
        train = pd.read_csv(train)
    if isinstance(test, basestring):
        test = pd.read_csv(test)

    # getting the cutoff values for x and y axis, using training set ONLY - because of the IMPORTANT ASSUMPTION -
    # TESTING SET IS SUBSET OF TRAINING SET IN TERMS OF X AND Y COORDINATES
    x_count, x_cutoff = np.histogram(train[x], bins = n)
    y_count, y_cutoff = np.histogram(train[y], bins = m)

    # transform cutoff values into step-wise tuples
    x_bin_tuple = [(floor, ceiling) for floor, ceiling in pairwise(x_cutoff)]
    y_bin_tuple = [(floor, ceiling) for floor, ceiling in pairwise(y_cutoff)]

    train_x_splits = split_df_rows_on_col_ranges(train, x, x_bin_tuple) # getting list of N bars based on x values for train
    test_x_splits = split_df_rows_on_col_ranges(test, x, x_bin_tuple) # getting list of N bars based on x values for test

    # within each bar (overall N) splitted based on x, there will be M splits based on y - each one is a grid
    trainDict = cut_y_bars_in_x_bar(train_x_splits, y, y_bin_tuple)
    testDict = cut_y_bars_in_x_bar(test_x_splits, y, y_bin_tuple)

    if outputFile:
        for key in trainDict:
            filename = 'train_' + 'x' + str(key[0]) + '_y' + str(key[1]) + '.csv'
            fullpath = os.path.join(train_output, filename)
            trainDict[key].to_csv(fullpath, index = False)
        for key in testDict:
            filename = 'test_' + 'x' + str(key[0]) + '_y' + str(key[1]) + '.csv'
            fullpath = os.path.join(test_output, filename)
            testDict[key].to_csv(fullpath, index = False)
    return (trainDict, testDict)
