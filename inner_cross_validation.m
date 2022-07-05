function [mean_metric, result] = inner_cross_validation(...
    predictors, ...
    target,...
    model_type, ...
    result, ...
    folds, ...
    outer_fold_iteration, ...
    hyperparameters ...
)
    n_samples = size(predictors, 1);
        
    % When the fold size cannot be divided into a natural number
    % will be using the floor of possibles folds and adding the final
    % records to the last fold.
    fold_size = floor(n_samples/folds);
    
    sum_metric = 0;
    
    % Fold cross-validation
    for f=1: folds
        fold_start = 1 + (f-1)*fold_size;
        fold_end   = f*fold_size;
        
        % the last fold will always hold the whole set of last items
        if f == folds
            fold_range = (fold_start:n_samples);
        else
            fold_range = (fold_start:fold_end);
        end
    
        % Obtain train split
        X_train = predictors;
        y_train = target;
        X_train(fold_range,:) = [];
        y_train(fold_range,:) = [];

        % Obtain test split
        X_test = predictors(fold_range,:);
        y_test = target(fold_range,:);
        
        model = fit_model(X_train, y_train, model_type, hyperparameters);

        % Evaluate model on test split
        y_pred = predict(model,X_test);
        metric = evaluate_metric(y_test, y_pred, model_type);
        sum_metric = metric + sum_metric;

        % save the results of each of the process iterations
        result = save_tunning_result( ...
            result, ...
            model_type, ...
            hyperparameters, ...
            outer_fold_iteration, ...
            f, ...
            model, ...
            metric ...
        );
        
    end
    % fprintf("Average metric in %f folds for %f: %f\n\n\n", ...
    %    folds, model_type, sum_metric/folds);

    mean_metric = sum_metric/folds;
end