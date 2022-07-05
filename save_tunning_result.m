function result = save_tunning_result( ...
    result, ...
    model_type, ...
    hyperparameters, ...
    outer_fold_iteration, ...
    inner_fold_iteration, ...
    model, ...
    metric ...
)
    % add all the hyperparameter tunning results to the results table
    result = [result;{
        model_type,...
        outer_fold_iteration, ...
        inner_fold_iteration, ...
        hyperparameters('KernelFunction'), ...
        hyperparameters('BoxConstraint'), ...
        hyperparameters('Epsilon'), ...
        hyperparameters('KernelScale'), ...
        hyperparameters('PolynomialOrder'), ...
        size(model.SupportVectors, 1), ...
        floor((size(model.SupportVectors, 1)/model.NumObservations)*100),...
        metric ...
    }];
end