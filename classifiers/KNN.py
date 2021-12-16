### KNN Classifier 
# Needs to be run from project base directory with 
#  python3 -m classifiers.[filename] (without the .py extension)

## Imports
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from consts import *
from metrics import *
from data import load_data
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


## KNN classifier training
def train_KNN(X, y, params=None, retrain=False, save_accuracy=True, save_model_flag=True):
    '''
        Input: 
            X - np array of training data
            y - np array of training labels
            (params) - python dictionary of parameters for KNN model
            (retrain) - whether or not to retrain or use saved model if false
        Output:
            Trained KNN model

        Function trains and returns a KNN model with given params
    '''

    # If no params given, make a blank one so defaults will be used
    if params is None:
        params = {}

    # get training parameters (defaults provided if not found in params)
    # --> others are not changed to try out in this project so default sklearn values used
    n_neighbors = params.get('n_neighbors', 1)      # number of neigbbors                   
    weights = params.get('weights', 'uniform')      # weight function used in prediction    
    algo = params.get('algo', 'auto')               # Algo for computing Nearest Ns         
    leaf_size = params.get('leaf_size', 30)         # used when algo = ball_tree or kd_tree
    p = params.get('p', 2)                          # l1 or l2 distance

    # directory for saving loading model
    save_dir = f"{SAVED_MODELS_DIR}/{KNN_MODEL_NAME}/"
    model_name = f"k={n_neighbors}, weights={weights}, algo={algo}, leaf_size={leaf_size}, p={p}, normalize={normalize}, train_proportion={TRAIN_PROPORTION}"
    save_model = save_dir + model_name

    # Check if we even need to train 
    if not retrain:
        # Check if we have a saved model, otherwise continue with training
        try:
            model = load(save_model)
            return model
        except:
            print('Saved KNN model not found -- retraining...')

    # Create model
    KNN_model = KNeighborsClassifier(n_neighbors=n_neighbors, 
                                      weights=weights, 
                                      algorithm=algo, 
                                      leaf_size=leaf_size, 
                                      p=p)  

    # Train model
    KNN_model.fit(X.reshape(-1, 3*32*32), y)

    # save model -- may need to make directory first -- only if we need to save (ie only during debugging)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if save_model_flag: 
        dump(KNN_model, save_model)

    # if need to predict here
    val_acc = None
    if save_accuracy:
        # get predictions on val set
        y_preds = predict_KNN(KNN_model, X_val) 

        # calculate accuracy on val set
        accuracy = calc_accuracy(y_val, y_preds)*100
        val_acc = accuracy

        # save accuracy
        save_file = save_dir + RESULTS_FILE
        with open(save_file, 'a') as fo:
            fo.write(f"{model_name} ==>{accuracy}\n")
    
    # return model
    return KNN_model, val_acc


## prediction function
def predict_KNN(KNN_model, X):
    '''
        Input: 
            X - data to predict with
        Output:
            predictions of trained KNN model 
    '''
    # get and return predictions
    preds = KNN_model.predict(X.reshape(-1,3*32*32))
    return preds


# ---------------    
#    MAIN CODE 
# ---------------

# initialise best KNN_model as None (will find during training or from saved file)
best_model = None
save_dir = f"{SAVED_MODELS_DIR}/{KNN_MODEL_NAME}/"
save_model = save_dir + "best_KNN_model"

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

    # loop through k values 1-10
    for k in tqdm(range(1,11)):
        params = {'n_neighbors': k}
        print(f"Training with k = {k}")

        # train and save accuracies
        KNN_model, val_acc = train_KNN(X_train, y_train, params=params, retrain=True, save_model_flag=False)
        acc_history.append(val_acc)

        # update 
        if best_acc < val_acc:
            best_acc = val_acc
            best_model = KNN_model
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
save_file = f"{SAVED_MODELS_DIR}/{KNN_MODEL_NAME}/{RESULTS_FILE}"
with open(save_file, 'r') as fo:
    # get all lines from file
    lines = fo.readlines()

    # loop through each line (model)
    for line in lines:
        # extract hyperparam and accuracy (skip test accuracy line)
        line = line.strip().split(", ")
        if (line[0] == '[TEST]'):
            continue
        hyperparam = int(line[0].split("=")[1])
        accuracy = float(line[-1].split(">")[-1])
        
        # store axes values
        x_axis.append(hyperparam)
        y_axis.append(accuracy)
      
# Plot 
plt.figure()
plt.title("Tuning KNN model\n(hyperparameter - number of neighbours)")
plt.ylabel("Validation accuracy")
plt.xlabel("K: Number of neighbours")
plt.plot(x_axis, y_axis, color='teal')
plt.savefig(f"{PLOTS_DIR}/KNN.pdf")
plt.clf()
print("Plotted validation")


## Get test accuracy -- either calculate if not yet done
if not values_calculated:
    # load best model
    if best_model is None:
        best_model = load(save_model)
    print("Best model loaded")

    # get predictions on test set
    y_preds = predict_KNN(best_model, X_test) 

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
    


        













    




