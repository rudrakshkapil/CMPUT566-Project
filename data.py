# File to load data:
#  expects dataset to be downloaded into dataset directory, 
#  which can be done by running sh download_cifar.sh 


## Imports
from os import name
import pickle
import numpy as np
from consts import *
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

## Function to unpickle data_batch_i file into labels and data -- from cifar-10 a
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

## Function to channel_wise_normalize 
def channelwise_normalize(X):
    # first convert from 0-255 to 0-1 (basically min-max normalization with 0 min) 
    X_norm = X / 255 

    # calculate channelwise means and stds
    means = np.mean(X_norm, axis=(0,2,3)).reshape(1, 3, 1, 1)
    stds = np.std(X_norm, axis=(0,2,3)).reshape(1, 3, 1, 1)
    
    # normalize
    X_norm = (X_norm - means) / stds

    # return
    return X_norm


## Function to return actual data in a format we can use
def load_data(train_proportion = TRAIN_PROPORTION, normalize=True):
    '''
        Inputs: 
            (train_proportion) - fraction of training set to be used for training 
                                (and the rest for validation set)
        Output: 
            X_train, y_train, X_test, y_test (Numpy arrays)

        This function unpickles the training batch files and concats 
        the required lists together before returning np arrays. 
        Also returns test data and labels. 
    '''
    # set seed
    np.random.seed(SEED)

    ## 1. Extract training data from all batches
    # all train data and labels
    train_data = []
    train_labels = []

    # loop through all train batches
    for idx in range(1,6):
        # unpickle to get batch data and labels
        batch_dict = unpickle(f'./dataset/cifar-10-batches-py/data_batch_{idx}')

        # extend all train data and labels by adding current batch data and labels to the end
        train_data.extend(batch_dict[b'data'])
        train_labels.extend(batch_dict[b'labels'])

    # Convert from lists to numpy arrays
    X_train = np.asarray(train_data).reshape(-1,3,32,32)
    y_train = np.asarray(train_labels)

    # shuffle numpy arrays
    X_train, y_train = shuffle(X_train, y_train)


    ## 2. Get a validation set
    num_train = int(train_proportion * len(X_train))

    # use the final num_train for validation
    X_val = X_train[num_train:]
    y_val = y_train[num_train:]

    # shorten training set
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]


    ## 3. Extract test data
    batch_dict = unpickle(f'./dataset/cifar-10-batches-py/test_batch')

    # convert to numpy arrays
    X_test = np.asarray(batch_dict[b'data']).reshape(-1,3,32,32)
    y_test = np.asarray(batch_dict[b'labels'])


    ## optional . Normalize 
    if normalize:
        X_train = channelwise_normalize(X_train) 
        X_val = channelwise_normalize(X_val)     
        X_test = channelwise_normalize(X_test)   


    ## 4. Return 
    return X_train, y_train, X_val, y_val, X_test, y_test


# debugging - DONE
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    print(np.bincount(y_val))
    print(np.bincount(y_train))

    plt.imshow(X_train[1].transpose(1,2,0))
    plt.title(str(y_train[0]))
    plt.show()
    
    




