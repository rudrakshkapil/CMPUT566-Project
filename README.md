# CMPUT566-Project
**Title**: A Study of Image Classification ML Algorithms, Including a Custom Implementation of CNNs

This repository contains the code for my final project for the UAlberta course CMPUT 466/566 Machine Learning.  

## Dataset
The dataset under consideration in this project is CIFAR-10. It can be downloaded by running the `download_cifar.sh` script. Running this script will save the dataset inside the `./dataset` directory.

## Classification Algorithms
`classifiers` contains python files for each image classification algorithm considered. To run any of these files, for example the k-nearest neighbours algorithm, run the follwing command from the base directory,

```
python3 -m classifiers.KNN
```

