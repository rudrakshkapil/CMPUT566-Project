### softmax Classifier 
# Needs to be run from project base directory with 
#  python3 -m classifiers.[filename] (without the .py extension)

## Imports
from numpy.core.fromnumeric import size
from numpy.lib.npyio import save
from data import load_data
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from consts import *
from metrics import *
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


## Softmax classifier training
def train_softmax(X, y, params=None, retrain=False, save_accuracy=True, save_model_flag=False):
    '''
        Input: 
            X - np array of training data
            y - np array of training labels
            (params) - python dictionary of parameters for softmax model
            (retrain) - whether or not to retrain or use saved model if false
        Output:
            Trained softmax model

        Function trains and returns a softmax model with given params
    '''
    # If no params given, make a blank one so defaults will be used
    if params is None:
        params = {}

    # get training parameters (defaults provided if not found in params)  https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions/52388406#52388406
    # --> others are not changed to try out in this project so default sklearn values used
    penalty = params.get('penalty', 'l2')        # l1, l2, or None regularization
    C = params.get('C', 1)                       # inverse of regularization strength
    solver = params.get('solver', 'saga')        # solver --> SAGA as suggested by scikit documentation https://scikit-learn.org/stable/modules/linear_model.html
    max_iter = params.get('max_iter', 100)       # max number of iterations

    # directory for saving loading model
    save_dir = f"{SAVED_MODELS_DIR}/{SOFTMAX_MODEL_NAME}/"
    model_name = f"penalty={penalty}, C={C}, solver={solver}, max_iter={max_iter}, normalize={normalize}, train_proportion={TRAIN_PROPORTION}"
    save_model = save_dir + model_name


    # Check if we even need to train 
    if not retrain:
        # Check if we have a saved model, otherwise continue with training
        try:
            model = load(save_model)
            return model
        except:
            print('Saved softmax model not found -- retraining...')

    # Create model
    softmax_model = LogisticRegression(multi_class='multinomial',
                                        penalty=penalty,
                                        C=C,
                                        solver=solver, 
                                        max_iter=max_iter)

    # Train model
    softmax_model.fit(X.reshape(-1, 3*32*32), y)

    # save model -- may need to make directory first
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if save_model_flag: 
        dump(softmax_model, save_model)

    # if need to predict here
    val_acc = None
    if save_accuracy:
        # get predictions on val set
        y_preds = predict_softmax(softmax_model, X_val) 

        # calculate accuracy on val set
        accuracy = calc_accuracy(y_val, y_preds)*100
        val_acc = accuracy

        # save accuracy
        save_file = save_dir + RESULTS_FILE
        with open(save_file, 'a') as fo:
            fo.write(f"{model_name} ==>{accuracy}\n")
    
    # return model
    return softmax_model, val_acc


## prediction function
def predict_softmax(softmax_model, X):
    '''
        Input: 
            X - data to predict with
        Output:
            predictions of trained softmax model 
    '''
    # get and return predictions
    preds = softmax_model.predict(X.reshape(-1, 3*32*32))
    return preds


# ---------------    
#    MAIN CODE 
# ---------------

# initialise best softmax_model as None (will find during training or from saved file)
best_model = None
save_dir = f"{SAVED_MODELS_DIR}/{SOFTMAX_MODEL_NAME}/"
save_model = save_dir + "best_softmax_model"


# Tune hyperparameter: k -- only if values are not already saved
values_calculated = True
if not values_calculated:
    # load data
    normalize = True
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(normalize=normalize)
    print("Data Loaded")

    # variables to store 
    acc_history = []
    best_acc = -1
    best_model = None

    # loop through ...
    C_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
    for C in tqdm(C_list):
        params = {'C': C}
        print(f"Training with C = {C}")

        # train and save accuracies
        softmax_model, val_acc = train_softmax(X_train, y_train, params=params, retrain=True, save_model_flag=False)
        acc_history.append(val_acc)

        # update 
        if best_acc < val_acc:
            best_acc = val_acc
            best_model = softmax_model
    print("Training all models done")
            
    # save best model
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dump(best_model, save_model)
    print("Saved best model")
    

## Plot validation 
# lists to store axes values
x_axis = []
y_axis = []

# open file and find accuracies (last value in each line)
save_file = f"{SAVED_MODELS_DIR}/{SOFTMAX_MODEL_NAME}/{RESULTS_FILE}"
with open(save_file, 'r') as fo:
    # get all lines from file
    lines = fo.readlines()

    # loop through each line (model)
    for line in lines:
        # extract hyperparam and accuracy (skip test accuracy line)
        line = line.strip().split(", ")
        if (line[0] == '[TEST]'):
            continue
        hyperparam = line[1].split("=")[1]
        accuracy = float(line[-1].split(">")[-1])
        
        # store axes values
        x_axis.append(hyperparam)
        y_axis.append(accuracy)
      
# Plot 
plt.figure()
plt.title("Tuning Softmax Regression model\n(hyperparameter - inverse of regularization strength)")
plt.ylabel("Validation accuracy")
plt.xlabel("C: Inverse of regularization strength")
plt.xticks(size=6)
plt.bar(x_axis, y_axis, color='orange', width=0.8)
plt.ylim(35, 40)
plt.savefig(f"{PLOTS_DIR}/Softmax.pdf")
plt.clf()
print("Plotted validation")


## Get test accuracy -- either calculate if not yet done
if not values_calculated:
    # load best model
    if best_model is None:
        best_model = load(save_model)
    print("Best model loaded")

    # get predictions on test set
    y_preds = predict_softmax(best_model, X_test) 

    # calculate accuracy on test set
    test_accuracy = calc_accuracy(y_test, y_preds)*100

    # save to file
    with open(save_file, 'a') as fo:
        fo.write(f"[TEST], {test_accuracy}\n")
    print(f"Test accuracy saved in file => {test_accuracy}%")

# or find from file
else:
    with open(save_file, 'r') as fo:
        # get all lines from file
        lines = fo.readlines()

        for line in lines:
            # find [TEST] line
            line = line.strip().split(", ")
            if line[0] != '[TEST]':
                continue
                
            # print accuracy
            print(f"Test accuracy read from file => {line[-1]}%")
    










    




