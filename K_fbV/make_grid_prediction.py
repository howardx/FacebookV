# this method is the task to be parallelized in multiple processes
# It uses xgb model for a single grid block and generate a dictionary of
#{(x,y) : xgb model prediction result for testing set (pd.DF)}

import xgboost as xgb
import numpy as np
import pandas as pd

import topK_prob as tp

import os

def predict_output(coord_tup, test_grid, model_dict, unique_class_dict, prediction_dict,
                   test_row_id, feature_list, saveFile = False, outputPath = None):
    rowID = test_grid[coord_tup][test_row_id].tolist() # ignore existing pandas indicies
    grid = test_grid[coord_tup][feature_list]

    test_X = xgb.DMatrix( grid.as_matrix() )
    preds_prob = model_dict[coord_tup].predict(test_X)

    sortedProbIdx = np.argsort(preds_prob)
    topKprob_idx = tp.get_top_k_idx(sortedProbIdx, 3)

    unique_class = unique_class_dict[coord_tup]
    predictions = unique_class[topKprob_idx]

    df_predictions = pd.DataFrame(predictions, index = None)
    df_predictions.insert(0, test_row_id, rowID)

    prediction_dict[coord_tup] = df_predictions

    if saveFile:
	filename = 'x' + str(coord_tup[0]) + '_y' + str(coord_tup[1]) + '_predict.csv'
	fullpath = os.path.join(outputPath, filename)
        df_predictions.to_csv(fullpath, index = False)
