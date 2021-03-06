%% OVERVIEW SCRIPT TO SHOW HOW TO TRAIN NETWORK
% feedforward neural network.

%% DEFINE THE SEARCH SPACE
% network 3,2 is 3 nodes in first layer, 1 nodes in second layer.
% network 4,2 is 4 nodes in first layer, 2 nodes in second layer.
% etc.
topologies = [3 1; ...
              3 2; ...
              4 2; ...
              4 3];
              
%% GENERATE TRAINING AND TEST SET

% one of matlab's data
[Xtrain,Ytrain] = vinyl_dataset;
Ytrain = Ytrain > -20;

%% DEFINE NUMBER OF NETWORKS TO TRAIN PER FOLD
N = 10; % ten networks, you'll find variance because of random init.

%% LOOP THROUGH EACH TOPOLOGY AND TRAIN
for kkk = 1:length(topologies)
    
    % GRAB THE TOPOLOGY TO EVALUATE
    nodes = topologies(kkk,:);
    
    % FYI - THIS IS THE NUMBER OF LAYERS IN THE NETWORK
    layers = size(nodes,2);
    
    % TRAIN ON GPU
    % gpuSet = true;
    
    % INITIAlIZE THE RANDOM SEED GENERATOR FOR REPRODUCIBILITY
    rng('default')
    
    % INITIALIZE THE NEURAL NETWORK
    net = initMyNetwork(nodes,'traingdx');
    
    % CHECK THE NUMBER OF WEIGHTS TO TRAIN
    net.numWeightElements
    
    %visualize to make sure it's correct
    view(net)
    
    % DONT FORGET TO CLEAR OUT PAST DATA
    clear predTrain
    clear predTest
    
    % TRAIN i MODELS (FOR MORE ROBUST RESULTS DUE TO RANDOM INIT)
    for i = 1:N
        
        % TRAIN
        % !!! observations are in columns, and features in row !!!
        nettr = train(net,Xtrain,Ytrain);
        
        % PREDICT
        predTrain(i,:) = nettr(Xtrain);
        % predTest(i,:) = nettr(Xtest);
    end

end