% Task 2 
% Use Matlab%s built-in fitcsvm/fitrsvm to train models, now using RBF and Polynomial
% kernels. Write an inner-fold cross-validation routine for finding the optimal
% hyperparameters. For classification, these are C and the kernel parameters: 'KernelScale',
% sigma for RBF and 'PolynomialOrder', q for the polynomial kernel. For regression, you
% have to set 'Epsilon' in addition. Do not use the automatic parameter optimisation
% methods. You must write your own, using only basic functions (although you can
% compare against that if you wish to check if you%re doing the right thing). You may
% want to do these tests on a subset of the data if training is too slow.
% Report for each model you trained how many support vectors were selected, both in
% absolute terms and in terms of a % of the training data available.

% For both datasets: Classification and Regression (/)
% fitcsvm/fitrsvm (X)
% Implement a inner-fold cross-validation routine.
%   Nested cross-validation
%   https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
% Using RBF and Polynomial kernel
% Parameters to optimize: (X)
%   Classification
%       C 
%       KernelScale sigma for RBF
%       PolynomialOrder q for Polynomial Kernel
%   Regression
%       C
%       Epsilon
%       KernelScale sigma for RBF
%       PolynomialOrder q for Polynomial Kernel
% Report in each trained model how many supports were selected
%   In absolute terms
%   In % of the training data available
clear all; close all; clc;
load('data/rice_data.mat');
load('data/concrete_data.mat');

% Get the table where the results of the nested-cross validation will be
% stored
result = get_result_table();

% Hyperparameters to optimize
% The possible range of values to optimize was "calculated" using
% this function:
% {fitcsvm|fitrsvm}(X,Y,'OptimizeHyperparameters','auto', ...
%    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
%    'expected-improvement-plus'));

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
    containers.Map({'dummy'},{0}),... % As the linear kernel doesn't have any intrinsic hyper-parameter,
                                  ... % we are just adding a dummy parameter to use the same grid-search
                                  ... % function implemented for gaussian and polynomial kernels
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

% See the results group by all the iterations made
% Regression
inter = sortrows(grpstats(result((strcmp(result.Kernel, 'gaussian') & strcmp(result.Type, 'regression')) ,:),{'Type','Kernel','C','Epsilon','KernelScale'}), 'mean_Percentaje_support_vectors_kept', 'ascend');
sortrows(inter(:,{'Type','Kernel','C','Epsilon','KernelScale','mean_Support_vectors', 'mean_Percentaje_support_vectors_kept', 'mean_Metric'}), {'C','Epsilon','KernelScale'}, 'descend');

inter = sortrows(grpstats(result((strcmp(result.Kernel, 'polynomial') & strcmp(result.Type, 'regression')) ,:),{'Type','Kernel','C','Epsilon','PolynomialOrder'}), 'mean_Percentaje_support_vectors_kept', 'ascend');
sortrows(inter(:,{'Type','Kernel','C','Epsilon','PolynomialOrder','mean_Support_vectors', 'mean_Percentaje_support_vectors_kept', 'mean_Metric'}), {'C','Epsilon','PolynomialOrder'}, 'descend');

inter = sortrows(grpstats(result((strcmp(result.Kernel, 'linear') & strcmp(result.Type, 'regression')) ,:),{'Type','Kernel','C','Epsilon'}), 'mean_Percentaje_support_vectors_kept', 'ascend');
sortrows(inter(:,{'Type','Kernel','C','Epsilon','mean_Support_vectors', 'mean_Percentaje_support_vectors_kept', 'mean_Metric'}), {'C','Epsilon'}, 'descend');

% classification
inter = sortrows(grpstats(result((strcmp(result.Kernel, 'gaussian') & strcmp(result.Type, 'classification')) ,:),{'Type','Kernel','C','KernelScale'}), 'mean_Percentaje_support_vectors_kept', 'ascend');
sortrows(inter(:,{'Type','Kernel','C','KernelScale','mean_Support_vectors', 'mean_Percentaje_support_vectors_kept', 'mean_Metric'}), {'C','KernelScale'}, 'descend');

inter = sortrows(grpstats(result((strcmp(result.Kernel, 'polynomial') & strcmp(result.Type, 'classification')) ,:),{'Type','Kernel','C','PolynomialOrder'}), 'mean_Percentaje_support_vectors_kept', 'ascend');
sortrows(inter(:,{'Type','Kernel','C','PolynomialOrder','mean_Support_vectors', 'mean_Percentaje_support_vectors_kept', 'mean_Metric'}), {'C','PolynomialOrder'}, 'descend');

inter = sortrows(grpstats(result((strcmp(result.Kernel, 'linear') & strcmp(result.Type, 'classification')) ,:),{'Type','Kernel','C'}), 'mean_Percentaje_support_vectors_kept', 'ascend');
sortrows(inter(:,{'Type','Kernel','C','mean_Support_vectors', 'mean_Percentaje_support_vectors_kept', 'mean_Metric'}), 'C', 'descend');

save('data/result_hyperparameter_tunning.mat','result');