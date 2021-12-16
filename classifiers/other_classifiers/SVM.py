### SVM Classifier 
# Needs to be run from project base directory with 
#  python3 -m classifiers.[filename] (without the .py extension)

## Imports
from data import load_data
from sklearn.svm import SVC
from joblib import dump, load
from consts import *
from metrics import *
import os
from sklearn.preprocessing import MinMaxScaler


## SVM classifier training
def train_SVM(X, y, params=None, retrain=False, save_accuracy=True):
    '''
        Input: 
            X - np array of training data
            y - np array of training labels
            (params) - python dictionary of parameters for SVM model
            (retrain) - whether or not to retrain or use saved model if false
        Output:
            Trained SVM model

        Function trains and returns a SVM model with given params
    '''

    

    # If no params given, make a blank one so defaults will be used
    if params is None:
        params = {}

    # get training parameters (defaults provided if not found in params)
    # --> others are not changed to try out in this project so default sklearn values used
    reg = params.get('reg', 1)
    kernel = params.get('kernel', 'rbf')
    gamma = params.get('gamma', 'scale')
    max_iter = params.get('max_iters', 100)


    # directory for saving loading model
    save_dir = f"{SAVED_MODELS_DIR}/{SVM_MODEL_NAME}/"
    model_name = f"reg={reg}, kernel={kernel}, gamma={gamma}, max_iter={max_iter}, normalize={normalize}, train_proportion={TRAIN_PROPORTION}"
    save_model = save_dir + model_name

    # Check if we even need to train 
    if not retrain:
        # Check if we have a saved model, otherwise continue with training
        try:
            model = load(save_model)
            return model
        except:
            print('Saved SVM model not found -- retraining...')

    # Create model
    SVM_model = SVC(C=reg, kernel=kernel, gamma=gamma, max_iter=max_iter)
  
    # flatten X
    X_new = X.reshape(-1, 3*32*32)

    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)

    # Train model
    SVM_model.fit(X_new, y)

    # save model -- may need to make directory first
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dump(SVM_model, save_model)

    # if need to predict here
    if save_accuracy:
        # get predictions on val set
        y_preds = predict_SVM(SVM_model, X_val) 

        # calculate accuracy on val set
        accuracy = calc_accuracy(y_val, y_preds)*100

        # save accuracy
        save_file = save_dir + RESULTS_FILE
        with open(save_file, 'a') as fo:
            fo.write(f"{model_name} => {accuracy}%\n")
    
    # return model
    return SVM_model


## prediction function
def predict_SVM(SVM_model, X):
    '''
        Input: 
            X - data to predict with
        Output:
            predictions of trained SVM model 
    '''
    # get and return predictions
    scaler = MinMaxScaler()
    X_new = X.reshape(-1, 3*32*32)
    X_new = scaler.fit_transform(X_new)
    preds = SVM_model.predict(X_new)
    return preds


# ---------------    
#    MAIN CODE 
# ---------------

# load data
normalize=False
X_train, y_train, X_val, y_val, X_test, y_test = load_data(normalize=normalize)

# train
SVM_model = train_SVM(X_train, y_train, retrain=True, params={'kernel':'rbf'})
print("SVM model trained/loaded")

# get predictions on val set
y_preds = predict_SVM(SVM_model, X_val) 

# calculate accuracy on val set
print(f"SVM accuracy => {calc_accuracy(y_val, y_preds)*100}%")


# TODO: Hyperparam tuning - determine best value of k and copy that model into a new one called best











    




