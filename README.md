# ml-nnet
Code to train a feedforward nnet with MATLAB. There are two scripts:

- **initMyNetwork.m**
- **nnet_script.m**

**initMyNetwork.m** is a function that will initialize a feed forward neural network according the parameters of the network size (layers and nodes) you provide it. Cost function (mse), node transform (tansig), output layer (softmax), training split, epochs, etc. are all hard-coded within **initMyNetwork.m**.

```
% defines a 2-layer neural network, with 3 nodes in the first layer, and 2 in the second.
hiddenSizes = [3 2];

% defines the training algorithm (stochastic gradient descent)
trainFcn = 'trainscg';

% returns initalized neural network object
[net] = initMyNetwork(hiddenSizes,trainFcn)
```

**nnet_script.m** is an example script that shows how to (1) define the neural network, (2) train it, and (3) evaluate it.

Tested on MATLAB 2016a and 2017a 
