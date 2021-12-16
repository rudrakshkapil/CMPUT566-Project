### CNN Classifier - custom Implementation
# Needs to be run from project base directory with 
#  python3 -m classifiers.[filename] (without the .py extension)

## Imports
from data import load_data
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import dump, load
import joblib
from consts import *
from metrics import *
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from classifiers.custom_CNN_utils.layers import *


## CNN classifier training
def train_CNN(X, y, params=None, retrain=False, save_accuracy=True, save_model_flag=False):
    '''
        Input: 
            X - np array of training data
            y - np array of training labels
            (params) - python dictionary of parameters for CNN model
            (retrain) - whether or not to retrain or use saved model if false
        Output:
            Trained CNN model

        Function trains and returns a custom CNN model with given params
    '''

    # If no params given, make a blank one so defaults will be used
    if params is None:
        params = {}

    # get training parameters (defaults provided if not found in params)
    # --> others are not changed to try out in this project so default sklearn values used
    lr = params.get('lr', 1e-4)
    batch_size = params.get('batch_size', 64) 
    epochs = params.get('epochs', 1)
    p = params.get("p", 0.1)


    # directory for saving loading model
    save_dir = f"{SAVED_MODELS_DIR}/{CUSTOM_CNN_MODEL_NAME}/"
    model_name = f"lr={lr}, batch_size={batch_size}, epochs={epochs}, p={p}, normalize={normalize}, train_proportion={TRAIN_PROPORTION}"
    save_model = save_dir + model_name

    # Check if we even need to train 
    if not retrain:
        # Check if we have a saved model, otherwise continue with training
        try:
            model = load(save_model)
            return model
        except:
            print('Saved CNN model not found -- retraining...')

    # 
    c_1 = 32
    c_2 = 64
    c_3 = 96
    c_4 = 64

    # Create model using custom layers
    CNN_model = nn.Sequential(
        # first conv-relu-batch-dropout unit
        custom_Conv2d(3, c_1, (5,5), padding=2),
        custom_ReLU(),
        custom_BatchNorm2d(c_1),
        custom_Dropout(p=p),

        # second conv-relu-batch-dropout unit
        custom_Conv2d(c_1, c_2, (3,3), padding=1),
        custom_ReLU(),
        custom_BatchNorm2d(c_2),
        custom_Dropout(p=p),

        # third conv-relu-batch-dropout unit
        custom_Conv2d(c_2, c_3, (3,3), padding=1),
        custom_ReLU(),
        custom_BatchNorm2d(c_3),
        custom_Dropout(p=p),

        # fourth conv-relu-batch-dropout unit
        custom_Conv2d(c_3, c_4, (3,3), padding=1),
        custom_ReLU(),
        custom_BatchNorm2d(c_4),
        custom_Dropout(p=p),

        # final linear layers -- Flatten is trivial, so just using nn. implementation
        nn.Flatten(),
        custom_Linear(c_4*32*32, 4096),
        custom_Linear(4096,10)
    )
    CNN_model.to(device) # use GPU

  
    ## Train model
    # convert data into dataset format and then create loaders
    X_temp = torch.from_numpy(X.astype(np.float32)).reshape(-1,3,32,32).to(device)
    y_temp = torch.from_numpy(y).to(device)
    train_data = torch.utils.data.TensorDataset(X_temp, y_temp)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # optimizer
    optimizer = optim.Adam(CNN_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # loss stats
    loss_history = []
    running_loss = 0.0

    # training loop(s)
    for epoch in tqdm(range(epochs), position=0, leave=True):
        for t, (batch_x, batch_y) in tqdm(enumerate(train_loader), position=0, leave=True, total=len(X)/batch_size):
            # put model into training mod
            CNN_model.train()

            # zero out gradients
            optimizer.zero_grad()

            # forward pass
            scores = CNN_model(batch_x)
            loss = criterion(scores, batch_y)
            loss_history.append(loss)

            # backward pass + optimize
            loss.backward()
            optimizer.step()

            # print 
            running_loss += loss.item()
            if t % 100 == 99:
                print('Iteration %d, loss = %.4f' % (t, running_loss/100))
                running_loss = 0

                # get predictions on val set 
                y_preds = predict_CNN(CNN_model, X_val) 

                # calculate accuracy on val set
                acc = calc_accuracy(y_val, y_preds)*100

                print("Validation accuracy => ", acc)
                

    # save model -- may need to make directory first
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if save_model_flag:
        dump(CNN_model, save_model)

    # if need to predict here
    val_acc = None
    if save_accuracy:
        # get predictions on val set 
        y_preds = predict_CNN(CNN_model, X_val) 

        # calculate accuracy on val set
        accuracy = calc_accuracy(y_val, y_preds)*100
        val_acc = accuracy

        # save accuracy
        save_file = save_dir + RESULTS_FILE
        with open(save_file, 'a') as fo:
            fo.write(f"{model_name} ==>{accuracy}\n")
    
    # return model
    return CNN_model, val_acc, loss_history


## prediction function
def predict_CNN(CNN_model, X):
    '''
        Input: 
            X - data to predict with
        Output:
            predictions of trained CNN model 
    '''
    # Put into eval mode
    CNN_model.eval()

    # convert to correct data format (don't need y, so use random tensor)
    X_new = torch.from_numpy(X.astype(np.float32)).to(device)
    scores = CNN_model(X_new.reshape((-1,3,32,32)))
    preds = scores.argmax(dim=1).cpu().numpy()   # get largest prob as prediction
    return preds


# ---------------    
#    MAIN CODE 
# ---------------


# initialise best CNN_model as None (will find during training or from saved file)
best_model = None
best_loss_history = None
save_dir = f"{SAVED_MODELS_DIR}/{CUSTOM_CNN_MODEL_NAME}/"
save_model = save_dir + "best_custom_CNN_model"

# GPU:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# load data
normalize = True 
X_train, y_train, X_val, y_val, X_test, y_test = load_data(normalize=normalize)
print("Data Loaded")

# Tune hyperparameter: n_channels
values_calculated = True
if not values_calculated:
    # variables to store 
    acc_history = []
    best_acc = -1
    best_model = None
    best_loss_history = None

    # only one p value, as determined by validation of pytorch CNN
    dropout_p_list = [0.25]   
    for p in tqdm(dropout_p_list):
        params = {'p': p}
        print(f"Training with p = {p}")

        # train and save accuracies
        CNN_model, val_acc, loss_history = train_CNN(X_train, y_train, params=params, retrain=True, save_model_flag=False)
        acc_history.append(val_acc)

        # update 
        if best_acc < val_acc:
            best_acc = val_acc
            best_model = CNN_model
            best_loss_history = loss_history
    print("Training all models done")
            
    # save best model and loss history for plotting later
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dump(best_model, save_model)
    print("Saved best model")
    dump(loss_history, save_model+"_loss_history")
    

## Plot validation 
# lists to store axes values
x_axis = []
y_axis = []

# open file and find accuracies (last value in each line)
save_file = f"{SAVED_MODELS_DIR}/{CUSTOM_CNN_MODEL_NAME}/{RESULTS_FILE}"
with open(save_file, 'r') as fo:
    # get all lines from file
    lines = fo.readlines()

    # loop through each line (model)
    for line in lines:
        # extract hyperparam and accuracy (skip test accuracy line)
        line = line.strip().split(", ")
        if (line[0] == '[TEST]'):
            continue
        hyperparam = line[3].split("=")[1]
        accuracy = float(line[-1].split(">")[-1])
        
        # store axes values
        x_axis.append(hyperparam)
        y_axis.append(accuracy)
      
# Plot 
if not os.path.isdir(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# no need to plot validation
if False:
    plt.figure()
    plt.title("Tuning CNN model\n(hyperparameter - probability of an element to be zeroed)")
    plt.ylabel("Validation accuracy")
    plt.xlabel("p: probability of an element to be zeroed in dropout layer")
    plt.plot(x_axis, y_axis, color='purple')
    plt.savefig(f"{PLOTS_DIR}/custom_CNN_validation.pdf")
    plt.clf()
    print("Plotted validation accuracies for different p values")

# plot loss
loss_flag = True
if best_loss_history is None:
    try:
        best_loss_history = load(save_model+"_loss_history")
    except:
        print("history not found or need to use GPU to load saved_best_loss_history")
        loss_flag = False

if loss_flag:
    plt.figure()
    plt.title("Training loss for best value of p")
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.plot(range(1, len(best_loss_history)+1), best_loss_history, color='purple')
    plt.savefig(f"{PLOTS_DIR}/custom_CNN_loss.pdf")
    plt.clf()
    print("Plotted training loss")





## Get test accuracy -- either calculate if not yet done
values_calculated = True
if not values_calculated and device == torch.device('cuda:0'):
    # load best model (trained on GPU)
    if best_model is None:
        best_model = load(save_model)
    print("Best model loaded")

    X_temp = torch.from_numpy(X_test.astype(np.float32)).reshape(-1,3,32,32).to(device)
    y_temp = torch.from_numpy(y_test).to(device)
    test_data = torch.utils.data.TensorDataset(X_temp, y_temp)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    ## get predictions on test set
    # Put into eval mode
    best_model.eval()

    # loop through batches and calculate accuracy
    num_correct = 0

    for t, (batch_x, batch_y) in tqdm(enumerate(test_loader), position=0, leave=True, total=10000/64):
        scores = best_model(batch_x.reshape((-1,3,32,32)))
        y_preds = scores.argmax(dim=1).cpu().numpy()     # get largest prob as prediction

        # sum correct
        batch_y = batch_y.cpu().numpy()
        num_correct += np.sum(batch_y == y_preds)


    # calculate accuracy on test set
    test_accuracy = num_correct/10000 *100

    # save to file
    with open(save_file, 'a') as fo:
        fo.write(f"[TEST], {test_accuracy}\n")
    print(f"Test accuracy saved in file => {test_accuracy}%")

# or find from file
else:
    try:
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
    except:
        print("File to read test accuracy from not found")










    




