### trivial baseline Classifier - DONE
# Needs to be run from project base directory with 
#  python3 -m classifiers.[filename] (without the .py extension)

## Imports
from data import load_data
from consts import *
from metrics import *


## KNN classifier training
def train_baseline():
    '''
        Input: 
            None
        Output:
            None

        Trivial baseline is random guessing, so this function does nothing
        because there is no need to train such a model.
    '''
    return None


## prediction function
def predict_baseline(X, num_classes=10):
    '''
        Input: 
            X - data to predict with (here only used to get dimensions)
        Output:
            random predictions between 0 to num_classes-1    
    '''

    # get and return predictions
    preds = np.random.choice(num_classes, len(X))
    return preds


# load data
X_train, y_train, X_val, y_val, X_test, y_test = load_data(train_proportion=0.99)

# train
baseline_model = train_baseline()

# get predictions on val set
y_preds = predict_baseline(X_val, num_classes=10) 

# calculate accuracy on val set
print(f"Accuracy => {calc_accuracy(y_val, y_preds)}")














    




