% Task 1: Train SVMs with the linear kernel
clear all; close all; clc;
% Load in the datasets as concrete_data and rice_data
load('data/concrete_data.mat');
load('data/rice_data.mat');

% ------------------------ %
% --------- RICE --------- %
% ------------------------ %

fprintf("------ RICE ------\n");

X_rice = rice_data(:, 1:4);
Y_rice = rice_data(:, 5);

% Train a model on the whole dataset
Mdl_rice = fitcsvm(X_rice, Y_rice, 'KernelFunction','linear', 'BoxConstraint', 1);  

% Convert to array for plotting ease
X_rice_arr = table2array(X_rice);    
Y_rice_arr = table2array(Y_rice);

% Get the support vectors and plot
sv_rice = Mdl_rice.SupportVectors;
figure
gscatter(X_rice_arr(:,1),X_rice_arr(:,2),Y_rice_arr)                           
hold on
plot(sv_rice(:,1),sv_rice(:,2),'ko','MarkerSize',10)          
legend('0','1','Support Vector')
title("Rice (classification) support vectors");
hold off

% Estimate the error through 10-fold cross-validation
rice_params = containers.Map( ...
    {'KernelFunction', 'BoxConstraint', 'PolynomialOrder', 'KernelScale', 'Epsilon'}, ...
    {'linear',          1,              [],                 [],     []});

result = get_result_table();
[classification_rate, result] = inner_cross_validation( ...
    X_rice, ...
    Y_rice, ...
    'classification', ...
    result, ...
    10, ...
    1, ...
    rice_params ...
);
fprintf("Classification rate: %f\n", classification_rate);


% ------------------------ %
% ------- CONCRETE ------- %
% ------------------------ %

fprintf("------ CONCRETE ------\n");

% Load the data in
X_conc = concrete_data(:, 1:8);
Y_conc = concrete_data(:, 9);

% Define a range of values to test epsilon
eps_range = [1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2];
best_rmse = 999999;
best_eps = 0;

for eps=eps_range
    % Setup the hyperparameters to pass to fit_model
    concrete_params = containers.Map( ...
    {'KernelFunction', 'BoxConstraint', 'PolynomialOrder', 'KernelScale', 'Epsilon'}, ...
    {'linear',          1,              [],                 [],     eps});

    % Run 10-fold cross validation to get an average classification
    % rate for this value of epsilon.
    result = get_result_table();
    [rmse, result] = inner_cross_validation( ...
        X_conc, ...
        Y_conc, ...
        'regression', ...
        result, ...
        10, ...
        1, ...
        concrete_params ...
        );

    if rmse < best_rmse
        best_rmse = rmse;
        best_eps = eps;
    end
    fprintf("epsilon: %.2f - RMSE: %f\n", eps, rmse);
end

% Plot the support vectors of a model trained on the entire
% dataset.

% Train a model on the whole dataset
Mdl_conc = fitrsvm(X_conc, Y_conc, 'KernelFunction','linear', 'BoxConstraint', 1, 'Epsilon', best_eps);

% Convert to array for plotting ease
X_conc_arr = table2array(X_conc);    
Y_conc_arr = table2array(Y_conc);

% Plot
sv_conc = Mdl_conc.SupportVectors;
figure
gscatter(X_conc_arr(:,1),X_conc_arr(:,2),Y_conc_arr)                           
hold on
plot(sv_conc(:,1),sv_conc(:,2),'ko','MarkerSize',10)          
legend('0','1','Support Vector')
title("Concrete (regression) support vectors");
hold off