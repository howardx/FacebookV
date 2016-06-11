# this method is the task to be parallelized in multiple processes
# It trains xgb model for a single grid block and fills two dictionaries -
# {(x,y) : xgb model}
# {(x,y) : list of unique classes for that grid - in original labels}

import xgboost as xgb

import feature_predictor_split as fps
import tune_numRound_maxDepth as tnm

import factorize_predictor as fp

import os

def train_model(coord_tup, train_grid, param, model_dict, unique_class_dict, feature_list,
                predictor_name, dump_model_txt = False,
                model_txt_file = None, save_model = False, save_model_file = None,
                tree_max_depth = 5, leaf_mltpr = 1, tree_mltpr = 1):
    feature_list.append(predictor_name) # can only have 1 predictor
    grid = train_grid[coord_tup][feature_list]

    predictor_idx = grid.columns.get_loc(predictor_name)
    
    train, original_label = fp.factorize_predictor(grid, predictor_idx)
    param['num_class'] = len(original_label)

    num_round, param['max_depth'] = tnm.tune_numRound_maxDepth(param['num_class'], num_leaf_mltpr = leaf_mltpr)
    print "number of boosters in training: " + str(num_round)

    train_X, train_Y = fps.feature_predictor_split(train, predictor_idx)
    xg_train = xgb.DMatrix( train_X, label = train_Y)

    # specify traing (and testing) set for model training
    watchlist = [ (xg_train,'train') ]

    # fill {(x,y) : xgb_model}
    bst = xgb.train(param, xg_train, num_round, watchlist)
    model_dict[coord_tup] = bst

    # {(x,y) : list of unique classes for that grid - in original labels}
    unique_class_dict[coord_tup] = original_label

    if dump_model_txt:
	filename = 'x' + str(coord_tup[0]) + '_y' + str(coord_tup[1]) + '_model.txt'
	fullpath = os.path.join(model_txt_file, filename)
        bst.dump_model(fullpath)
    if save_model:
	filename = 'x' + str(coord_tup[0]) + '_y' + str(coord_tup[1]) + '.model'
	fullpath = os.path.join(save_model_file, filename)
        bst.save_model(fullpath)
