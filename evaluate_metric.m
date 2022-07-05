function metric = evaluate_metric( ...
    y_test, ...
    y_pred, ...
    model_type ...
)
    if model_type == "regression"
        % square root metric
        ABSE = table2array(y_test) - y_pred; % Absolute Error
        metric = sqrt(dot(ABSE',ABSE)); % Root Mean Square Error
    else
        % accuracy
        metric = (1-(sum(abs(table2array(y_test) - y_pred)) / size(y_test, 1)))*100;
    end
end