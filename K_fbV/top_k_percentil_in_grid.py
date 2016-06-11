import numpy as np
import pandas as pd

def top_K_percentile_pid(grid, k_p = 80, pid = 'place_id'):
    grid['pid_count'] = grid.groupby(pid)[pid].transform(pd.Series.value_counts)

    grid.sort_values(by = ['pid_count', pid], ascending = False, inplace = True)
    numRow = grid.shape[0]

    percentile_idx = int(0.01 * k_p * numRow) - 1
    if percentile_idx < 0:
        percentile_idx = 0 # in case the number of rows are too few

    # iloc() selects row by integer index, NOT labled index - for labeled index use loc()
    percentil_border_value = grid.iloc[[percentile_idx]][pid]

    # getting the percentil position place_id value, see if there were more than one instances
    # reason to use np.where() is because it returns row index, not row label from pandas.DataFrame
    percentil_border_chunk = np.where(grid[pid] == percentil_border_value.values[0])

    # np.where() always returns a tuple of arrays, because you can directly plug its result back in
    # to your original array as index and perform select
    percentil_border = max(percentil_border_chunk[0])

    grid.drop('pid_count', axis = 1, inplace = True)

    return grid[:percentil_border + 1] # pandas.DataFrame row slice is end exclusive
