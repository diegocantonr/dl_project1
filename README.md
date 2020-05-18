# EE-559 Deep Learning - Project 1 - Convolutional Neural Network and improvements for MNIST comparison

In this project, we implement multiple network architectures able to predict from a pair of images if the first digit is lesser or equal to the second and we look at the improvement that can be achieved by the use of weight sharing and auxiliary losses. 

## Prerequisites
Python 3.7 
PyTorch
Seaborn (only to plot the final boxplot)
Pandas (only to plot the final boxplot)

## Dataset 
The dataset consist of MNIST images, loaded from the dlc_practical_prologue.py file. The train set and test set consist of 1000 pairs of 14x14 grayscale images in addition to their target {0, 1} and classes {0, ..., 9}. 

