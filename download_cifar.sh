#!/bin/bash

# This file downloads the cifar-10 dataset and stores it inside ./dataset directory

# make and enter directory
mkdir dataset
cd dataset

# download zip, unzip, and delete the zip
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz

# echo
echo "Downloaded CIFAR-10 inside ./dataset directory"