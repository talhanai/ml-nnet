# ml-nnet
Code to train a feedforward nnet with MATLAB.

**initMyNetwork.m** is a function that will initialize a feed forward neural network according the parameters you provide it.

```
% defines a 2-layer neural network, with 3 nodes in the first layer, and 2 in the second.
hiddenSizes = [3 2];

% defines the training algorithm (stochastic gradient descent)
trainFcn = 'trainscg';

% returns initalized neural network object
[net] = initMyNetwork(hiddenSizes,trainFcn)
```

**nnet_script.m** is an example script that shows how to (1) define the neural network, (2) train it, and (3) evaluate it.
