import pandas as pd
import os

def merge_prediction_grids(predict_dict, sort_rowID = True, save_file = False, output_path = None):
    df_finalResult = pd.concat(predict_dict.values(), ignore_index = True)

    if sort_rowID:
        df_finalResult = df_finalResult.sort(['row_id'])

    if save_file:
	filename = os.path.join(output_path, 'submission.csv')
        df_finalResult.to_csv(filename, index = False) # no pandas index in CSV file
    return df_finalResult
