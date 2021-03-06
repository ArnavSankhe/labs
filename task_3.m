% Task 3
% Perform 10-fold cross-validation on the binary classification and regression tasks.
% Report results for linear, Gaussian, and polynomial kernels. Optimise hyper-parameters
% with the inner-cross-validation procedure developed in Task 2.

% Report on the classification rate for the classification problems and RMSE for the
% regression problems. You may once again opt to use less data (if your selected data is
% too large) for your evaluation.

clear all; close all; clc;
load('data/rice_data.mat');
load('data/concrete_data.mat');

% Running the hyper-parameter tunning process takes
% some time. We already ran the process with the hyper-parameters below
% for both datasets. The results were saved for later result report.
% You can use the next line to load the result of the hyper-parameter
% tunning process or you can run each function again if you need to change
% any of the hyper-parameters.
% load('data/result_hyperparameter_tunning.mat');

% Get the table where the results of the nested-cross validation will be
% stored
result = get_result_table();

% Hyperparameters to optimize
C = containers.Map(...
    {'0.001', '0.003', '0.005', '0.01', '0.03', '0.05','0.1', '0.3', '0.5','1', '3', '5'}, ...
    {0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5} ...
);

Epsilon = containers.Map(...
    {'0.01', '0.03', '0.05','0.1', '0.3', '0.5','1', '3', '5'}, ...
    {0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5} ...
);

Kernel = containers.Map(...
    {'linear', 'gaussian', 'polynomial'}, ...
    {
    containers.Map(),...
    containers.Map(...
        {'0.01', '0.03', '0.05','0.1', '0.3', '0.5','1', '3', '5'}, ... % Kernel Scale
        {0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5} ...
    ), containers.Map(...
        {'3', '2', '1'}, ... % polynomial order
        {3, 2, 1} ...
    )} ...
);

outer_folds = 10;
inner_folds = 2;

% Execute the nested cross_validation for the regression
% dataset and store the results
result = nested_cross_validation( ...
    concrete_data(:,1:8), ...
    concrete_data(:,9),...
    "regression", ...
    result, ...
    outer_folds, ...
    inner_folds, ...
    C, ...
    Epsilon, ...
    Kernel ...
);

% Execute the nested cross_validation for the classification
% dataset and store the results
result = nested_cross_validation( ...
    rice_data(:,1:4), ...
    rice_data(:,5),...
    "classification", ...
    result,...
    outer_folds, ...
    inner_folds, ...
    C, ...
    Epsilon, ...
    Kernel ...
);