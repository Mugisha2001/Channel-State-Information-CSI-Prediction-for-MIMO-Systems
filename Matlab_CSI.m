% CSI Prediction using LSTM in MATLAB
clc;
clear;
close all;

%% Generate Simulated CSI Data
rng(0); % Set random seed for reproducibility
dataLength = 1000; % Number of samples
timesteps = 10; % Number of timesteps for LSTM
noise = 0.5 * randn(1, dataLength); % Add Gaussian noise
csiData = sin(linspace(0, 20, dataLength)) + noise; % Simulated CSI

%% Prepare Dataset for LSTM
X = [];
y = [];
for i = 1:(dataLength - timesteps)
    X = [X; csiData(i:i+timesteps-1)]; % Extract sequences of length 'timesteps'
    y = [y; csiData(i+timesteps)]; % Target value (next value in the sequence)
end

% Normalize Data
X = normalize(X, 'range', [-1, 1]); % Normalize inputs to range [-1, 1]
y = normalize(y, 'range', [-1, 1]); % Normalize outputs to range [-1, 1]

% Reshape Data for LSTM (Samples x Timesteps x Features)
X = reshape(X, size(X, 1), timesteps, 1); % Convert X into 3D array: [samples, timesteps, features]

% Validate Dimensions
disp("Shape of X after reshaping:");
disp(size(X)); % Should be [990, 10, 1] (990 samples, 10 timesteps, 1 feature)

%% Split Data into Training and Testing Sets
splitRatio = 0.8; % 80% training, 20% testing
splitIdx = round(size(X, 1) * splitRatio); % Index for splitting data
XTrain = X(1:splitIdx, :, :); % Training data (80% of the samples)
yTrain = y(1:splitIdx); % Corresponding target values
XTest = X(splitIdx+1:end, :, :); % Testing data (20% of the samples)
yTest = y(splitIdx+1:end); % Corresponding target values

% Validate Training and Testing Dimensions
disp("Shape of XTrain:");
disp(size(XTrain)); % Should be [792, 10, 1] (792 training samples, 10 timesteps, 1 feature)
disp("Shape of yTrain:");
disp(size(yTrain)); % Should be [792, 1] (792 target values for training)

%% Define LSTM Network
layers = [
    sequenceInputLayer(1, 'Name', 'input') % Input layer expects sequences with 1 feature
    lstmLayer(50, 'OutputMode', 'last', 'Name', 'lstm') % LSTM layer with 50 units
    fullyConnectedLayer(1, 'Name', 'fc') % Fully connected layer for regression output
    regressionLayer('Name', 'output') % Regression layer for numeric prediction
];

% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ... % Number of training epochs
    'MiniBatchSize', 32, ... % Mini-batch size for training
    'InitialLearnRate', 0.001, ... % Initial learning rate
    'Verbose', true, ... % Display training progress
    'Plots', 'training-progress'); % Plot training progress

%% Train the LSTM Model
net = trainNetwork(XTrain, yTrain, layers, options);

%% Predict on Test Data
yPred = predict(net, XTest); % Get predictions on the test set

%% Evaluate Performance
mse = mean((yPred - yTest).^2); % Calculate Mean Squared Error between predicted and actual values
fprintf('Mean Squared Error: %.4f\n', mse);

%% Visualize Results
figure;
plot(yTest, 'b', 'DisplayName', 'Actual CSI'); % Plot actual CSI values
hold on;
plot(yPred, 'r--', 'DisplayName', 'Predicted CSI'); % Plot predicted CSI values
xlabel('Sample Index');
ylabel('CSI Value');
title('CSI Prediction using LSTM');
legend;
grid on;

