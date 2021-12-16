### Decision Tree Classifier 
# Needs to be run from project base directory with 
#  python3 -m classifiers.[filename] (without the .py extension)

## Imports
from numpy.lib.npyio import save
from data import load_data
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
from consts import *
from metrics import *
import os


## Decision Tree classifier training
def train_decision_tree(X, y, params=None, retrain=False, save_accuracy=True):
    '''
        Input: 
            X - np array of training data
            y - np array of training labels
            (params) - python dictionary of parameters for Decision Tree model
            (retrain) - whether or not to retrain or use saved model if false
        Output:
            Trained Decision Tree model

        Function trains and returns a Decision Tree model with given params
    '''

    # If no params given, make a blank one so defaults will be used
    if params is None:
        params = {}

    # get training parameters (defaults provided if not found in params)
    # --> others are not changed to try out in this project so default sklearn values used
    criterion = params.get('criterion', 'gini')                # function to measure the quality of a split
    splitter = params.get('splitter', 'best')                  # strategy used to choose the split at each node
    max_depth = params.get('max_depth', None)                  # max depth of tree (None --> until all)
    min_samples_split = params.get('min_samples_splitint', 20)  # minimum number of samples required to split an internal node
    min_samples_leaf = params.get('min_samples_leaf', 10)       # minimum number of samples required to be at a leaf node

    # directory for saving loading model
    save_dir = f"{SAVED_MODELS_DIR}/{DECISION_TREE_MODEL_NAME}/"
    model_name = f"criterion={criterion}, split={splitter}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, normalized={normalize}, train_proportion={TRAIN_PROPORTION}"
    save_model = save_dir + model_name

    # Check if we even need to train 
    if not retrain:
        # Check if we have a saved model, otherwise continue with training
        try:
            model = load(save_model)
            return model
        except:
            print('Saved Decision Tree model not found -- retraining...')

    # Create model
    decision_tree_model = DecisionTreeClassifier()
  
    # flatten X
    X = X.reshape(-1, 3*32*32)
  
    # Train model
    decision_tree_model.fit(X, y)

    # save model -- may need to make directory first
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dump(decision_tree_model, save_model)

    # if need to predict here
    if save_accuracy:
        # get predictions on val set
        y_preds = predict_decision_tree(decision_tree_model, X_val) 

        # calculate accuracy on val set
        accuracy = calc_accuracy(y_val, y_preds)*100

        # save accuracy
        save_file = save_dir + RESULTS_FILE
        with open(save_file, 'a') as fo:
            fo.write(f"{model_name} => {accuracy}%\n")
    
    # return model
    return decision_tree_model


## prediction function
def predict_decision_tree(decision_tree_model, X):
    '''
        Input: 
            X - data to predict with
        Output:
            predictions of trained Decision Tree model 
    '''
    # get and return predictions
    preds = decision_tree_model.predict(X.reshape(-1, 3*32*32))
    return preds


# ---------------    
#    MAIN CODE 
# ---------------

# load data
normalize = False
X_train, y_train, X_val, y_val, X_test, y_test = load_data(normalize=normalize)
print("Data loaded")

# train
decision_tree_model = train_decision_tree(X_train, y_train, retrain=False)
print("Decision Tree model trained/loaded")

# get predictions on val set
y_preds = predict_decision_tree(decision_tree_model, X_val) 

# calculate accuracy on val set
print(f"Decision Tree accuracy => {calc_accuracy(y_val, y_preds)*100}%")


# TODO: Hyperparam tuning - determine best value of k and copy that model into a new one called best











    




