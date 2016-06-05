# method to tune relation ship between num_round and max_depth
import math

def tune_numRound_maxDepth(num_class, max_depth = 5, num_leaf_mltpr = 1, num_tree_mltpr = 1):
    num_leafes = num_class * num_leaf_mltpr

    max_leaf_per_tree = math.pow(2, max_depth)

    num_tree = int ( (num_leafes / max_leaf_per_tree) * num_tree_mltpr )
    return (num_tree, max_depth)
