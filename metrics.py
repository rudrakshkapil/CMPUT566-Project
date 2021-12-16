## File containing code for classification metrics 
import numpy as np

# accuracy
def calc_accuracy(t, y):
    '''
        Input:
            t - target labels,
            y - predicted labels
        Output:
            Classification accuracy
    '''
        
    return np.mean(t==y)