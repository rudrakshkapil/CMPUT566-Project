# CMPUT566-Project
**Title**: A Study of Image Classification ML Algorithms, Including a Custom Implementation of CNNs

This repository contains the code for my final project for the UAlberta course CMPUT 466/566 Machine Learning. A version of the project report has been uplaoded to this repository, however the final version is the one submitted through eClass.  

## Organization

### Classification Algorithms
`classifiers` contains python files for each image classification algorithm considered. To run any of these files, for example the k-nearest neighbours algorithm, run the follwing command from the base directory,

```
python3 -m classifiers.KNN
```

For training both the PyTorch and custom CNN models, I have used the `running_with_GPU.ipynb` notebook to use the GPU provided by Google Colab. 

### Saved Models
The trained models whose accuracies are quoted in the accompanying project report can be downlaoded from this [link](https://drive.google.com/drive/folders/1DFSq8fYcm0zlDnGwbsVPN2esuh-hLncc?usp=sharing).
...


### Dataset
The dataset under consideration in this project is CIFAR-10. It can be downloaded by running the `download_cifar.sh` script. Running this script will save the dataset inside the `./dataset` directory.
The python file `data,py` contains utility code for loading and preprocessing the dataset. This code is implicitly called from the classifier python files.


### Evaluation
`metrics.py` contains common utility code for evaluating the performance of all ML models. 


### Common Constants
`consts.py` contains constants used across all python files, to ensure consistency in naming, locations, etc. 

