# EE-559 Deep Learning - Project 1 - Convolutional Neural Network and improvements for MNIST comparison

In this project, we implement multiple network architectures able to predict from a pair of images if the first digit is lesser or equal to the second and we look at the improvement that can be achieved by the use of weight sharing and auxiliary losses. 

## Prerequisites
Python 3.7 <br/>
PyTorch <br/>
Seaborn (only to plot the final boxplot) <br/>
Pandas (only to plot the final boxplot) <br/>

## Dataset 
The dataset consist of MNIST images, loaded from the dlc_practical_prologue.py file. The train set and test set consist of 1000 pairs of 14x14 grayscale images in addition to their target {0, 1} and classes {0, ..., 9}. 

# Organisation of the repository

```
|
|   README.md                                       > README of the project  
|   CNN.py                                          > definition of baseline convolutional neural net
|   WS.py                                           > definition cnn with weight sharing
|   WS_AL.py                                        > definition cnn with weight sharing and auxiliary loss
|   dlc_practical_prologue.py                       > helper file provided by François Fleuret for the course
|   helper.py                                       > contains functions to plot, generate data and train model
|   test.py                                         > runs our best model 
|   tuning.py                                       > tuning (grid search) of model
|
```  
