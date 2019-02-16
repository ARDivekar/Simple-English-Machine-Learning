# Notation and terminology for feedforward networks

When you deal with feedforward networks, you can often lose track of which exact neuron or layer is being referred to at any point in time. 

For that reason, I use the following terminology:

* The network is split into $$L$$ layers, $$W_1, W_2, \dots, W_l, \dots, W_{L}$$. These are all **hidden** layers. The input vector/tensor to the network is not considered a layer, neither is the output vector/tensor. 
  * The main property of a layer is that it has trainable weights attached to it: the layer "owns" the weights.
* For inputs to a layer $$L_i$$, $$X_i$$or $$x_i$$ are used interchangeably.



 

