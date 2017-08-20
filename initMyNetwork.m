function [net] = initMyNetwork(hiddenSizes,trainFcn)
% [net] = initMyNetwork(hiddenSizes,trainFcn)
% This script initializes a feedforward neural network.
%
%   Inputs:
%
%       nodes:  a vector defining the number of nodes in each layer.        
%       e.g. [3 2] is a 2 layer network with 3 nodes in the first and 
%       2 nodes in the second layer.
%
%       trainFcn: is the training algorithm for the neural network. It can
%       be  
%           'trainscg' - scaled conjugate gradient backprop.
%           'traingd'  - grad descent backprop.
%           'traingdx' - grad descent + momentum + adaptive lr backprop. 
%           'traingdm' - grad descent + momentum 
%           'traingda' - grad descent + adaptive lr backprop
%           'trainlm'  - Levenberg-Marquardt backprop.
%
%   Output:
%
%       net: is a neural network object. Look at 'help network' for more
%       details.
%
%   Configured to run until convergence else 1,000 epochs. Loss function is
%   'mse'.
%
% Authors: Tuka Alhanai and Mohammad Ghassemi, Oct 2016, Aug 2017
%
% Tested on MATLAB 2016a, 2017a


net = network;
net.name = 'NNET Classifier';

Nl = size(hiddenSizes,2)+1;
net.numLayers = Nl;
net.biasConnect = true(Nl,1);

%% This generates the fully connected network, given the sizes you specified
%if you want to see the network, please run 'view(net)' at the end of this
%code block.
[j,i] = meshgrid(1:Nl,1:Nl);
net.layerConnect = (j == (i-1));
net.outputConnect = [false(1,Nl-1) true];
for i=1:Nl
    if i == Nl
        net.layers{i}.name = 'Output';
        net.layers{i}.transferFcn = 'softmax';
        net.layers{i}.initFcn = 'initwb';
    else
        if (Nl == 2)
            net.layers{i}.name = 'Hidden';
        else
            net.layers{i}.name = ['Hidden ' num2str(i)];
        end
        net.layers{i}.size = hiddenSizes(i);
        net.layers{i}.transferFcn = 'tansig';
        net.layers{i}.initFcn = 'initnw';
    end 
end

net.numInputs = 1;
net.inputConnect(1,1) = true;

%%  Set Important Training parameters
net.performFcn = 'mse';
net.trainFcn = trainFcn;
net.trainParam.epochs = 1000;
net.trainParam.showWindow = 0;
net.trainParam.showCommandLine = 0;

%Uncomment this if you want the network to use everything for training.
%net.divideFcn = 'dividetrain';

%Set these values to determine the size of the training, validation and
%testing sets.
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio=0.0;

% EOF
