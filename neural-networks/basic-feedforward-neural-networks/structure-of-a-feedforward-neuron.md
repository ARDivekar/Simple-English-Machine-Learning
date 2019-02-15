# Structure of a Feedforward Neuron

A feedforward neuron is what comes to most people's minds when they think of neural networks.

This neuron is a vector of weights. It takes as input a vector from the previous layer/input layer. It computes the dot product of these two \(called an affine transform\) and then outputs a scalar value. The value it outputs is then transformed by an activation function into another scalar value, which is passed to the next layer/network output.

