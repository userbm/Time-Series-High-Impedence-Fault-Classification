# Time Series (High-Impedence-Fault) Classification
High Impedence Fault Classification is a time series classification problem. Model is 1D convolutional.

## Dataset
Each instance conists of 30 step recording of energy signal in the system.
The output is two class - fault or not.
1385 instances are used for training.
The data is not open sourced. 
The code can be easily used for other datasets without much modifications.

## Model
The model is a simple conv model with 4 layers of depthwise separable 1D convolutions followed by linear layers.
Implementation is done in Pytorch.

pytorch                   1.1.0

torchvision               0.3.0
