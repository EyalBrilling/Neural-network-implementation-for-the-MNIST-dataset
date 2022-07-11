# Neural network classifier implementation for the MNIST-dataset
Neural network implementation, labeling numbers handwritten pictures from 0 to 9 


## The Architecture
The Neural network is composed of 4 parts.
*****The input layer***** - 784 neurons corresponding to 784 pixels in the number PNG (28x28)

*****The first and second hidden layer***** - 128/64 neurons calculated by activating the sigmoid function on the weights * the pixels vector +bias

*****The output layer***** - 10 neurons corresponding to 10 tags (0-9) calculated by taking the second hidden layer output and activating the softMax function on it.

the most probable number guess from the neural network is taken as the chosen tag.

<p align="center"><img src="https://github.com/EyalBrilling/Neural-network-implementation-for-the-MNIST-dataset/blob/master/NN_PNG/NN_PNG.png" width="600" height="400" /></p>
