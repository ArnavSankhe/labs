function [best_hyperparameters, result] = grid_search(...
    predictors, ...
    target, ...
    model_type, ...
    result,...
    inner_folds, ...
    outer_fold_iteration,...
    C, ...
    Epsilon, ...
    Kernel ...
)
    hyperparameters = containers.Map(...
        {'KernelFunction', 'BoxConstraint', 'PolynomialOrder', 'KernelScale', 'Epsilon'}, ...
        {[],               [],               [],               [],      []} ...
    );

    best_hyperparameters = [];
    best_metric = 0;

    % For all values of C
    % c_keys = keys(C);
    c_val = values(C);
    for i = 1:length(C)
        % c_label = c_keys{i};
        hyperparameters('BoxConstraint') = c_val{i};
        
        % For all the kernels
        kernel_keys = keys(Kernel);
        kernel_val = values(Kernel);
        for j = 1:length(Kernel)
            kernel = kernel_keys{j};
            kernel_params = kernel_val{j};
            hyperparameters('KernelFunction') = kernel;
            
            % For each of the hyperarameters of the kernels
            % kernel_params_keys = keys(kernel_params);
            kernel_params_val = values(kernel_params);
            for m = 1:length(kernel_params)
                    % kernel_param_label = kernel_params_keys{m};
                    kernel_param = kernel_params_val{m};
                    
                    % set hyperparameter by kernel
                    if kernel == "polynomial"
                        hyperparameters('PolynomialOrder') = kernel_param;
                        hyperparameters('KernelScale') = NaN;
                    end
                    
                    if kernel == "gaussian"
                        hyperparameters('KernelScale') = kernel_param;
                        hyperparameters('PolynomialOrder') = NaN;
                    end

                    if kernel == "linear"
                        hyperparameters('KernelScale') = NaN;
                        hyperparameters('PolynomialOrder') = NaN;
                    end

                    % here implementation for gaussian
                    
                    % epsilon is a paramerter just for regression
                    % so we won't add this hyperparameter while fitting
                    % the classification model
                    if model_type == "regression"
                        % For all epsilon values
                        % ep_keys = keys(Epsilon);
                        ep_val = values(Epsilon);
                        for n = 1:length(Epsilon)
                            % ep_label = ep_keys{n};
                            epsilon = ep_val{n};
                            
                            % get the results of each of the models using
                            % inner cross validation
                            hyperparameters('Epsilon') = epsilon;

                            [metric, result] = inner_cross_validation( ...
                                predictors, ...
                                target, ...
                                model_type, ...
                                result,...
                                inner_folds, ...
                                outer_fold_iteration, ...
                                hyperparameters ...
                            );
                            
                            % save the best set of hyperparameters
                            if metric > best_metric
                                best_hyperparameters = hyperparameters;
                                best_metric = metric;
                            end
                        
                        end
                    else
                        hyperparameters('Epsilon') = NaN;
                        [metric, result] = inner_cross_validation( ...
                            predictors, ...
                            target, ...
                            model_type, ...
                            result,...
                            inner_folds, ...
                            outer_fold_iteration, ...
                            hyperparameters ...
                        );

                        % save the best set of hyperparameters
                        if metric > best_metric
                            best_hyperparameters = hyperparameters;
                            best_metric = metric;
                        end
                    end
            end
    end

    
end