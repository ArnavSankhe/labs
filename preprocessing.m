% Preprocessing and data exploration

% We will define an outlier as: a value that is more than 
% three scaled median absolute deviations (MAD) away from the median.

% https://archive.ics.uci.edu/ml/datasets/Rice+%28Cammeo+and+Osmancik%29
% https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

% Preprocessing stages:
% 1. Data exploration
% 2. Missing values
% 3. Outliers
% 4. Normalization
% 5. Correlations
clear all; close all; clc;

rice_data = readtable(('./data/Rice_Osmancik_Cammeo_Dataset.csv'), ...
    'HeaderLines', 0, 'ReadVariableNames', true);
concrete_data = readtable(('./data/Concrete_Dataset.csv'),'HeaderLines',0, ...
    'ReadVariableNames', true);

% ++++++++++++++++++++++++++++++++++++++++++++
% ++++++ Rice Dataset Preprocessing ++++++++++
% ++++++++++++++++++++++++++++++++++++++++++++

% --- Response-Binarization ---
unique(rice_data.CLASS);
% Cammeo->0
% Osmancik->1
is_osmancik = ismember(rice_data.CLASS,('Osmancik'));
rice_data(:,9) = array2table(double(is_osmancik));
rice_data = removevars(rice_data,{'CLASS'});
rice_data.Properties.VariableNames([8])={'CLASS'};

% Data size (3810 x 8)
size(rice_data);

% Missing values-0 values missing
fprintf("Rice Data Missing Values")
rice_data(any(ismissing(rice_data),2),:)

% --- Area-Data exploration ---
figure, boxplot(rice_data.AREA), title("Rice Area");
figure, histogram(rice_data.AREA), title("Rice Area");
% Outliers-3 values
rice_data(isoutlier(rice_data.AREA),:);

% --- Minoraxis-Data exploration ---
figure, boxplot(rice_data.MINORAXIS), title("Rice Minoraxis");
figure, histogram(rice_data.MINORAXIS), title("Rice Minoraxis");
% Outliers-30 values
rice_data(isoutlier(rice_data.MINORAXIS),:);

% --- Extent-Data exploration ---
figure, boxplot(rice_data.EXTENT), title("Rice Extent");
figure, histogram(rice_data.EXTENT), title("Rice Extent");
% Outliers-0 values
rice_data(isoutlier(rice_data.EXTENT),:);

% --- Convex Area-Data exploration ---
figure, boxplot(rice_data.CONVEX_AREA), title("Rice Convex Area"); 
figure, histogram(rice_data.CONVEX_AREA), title("Rice Convex Area");
% Outliers-0 values
rice_data(isoutlier(rice_data.CONVEX_AREA),:);

% --- Eccentricity-Data exploration ---
figure, boxplot(rice_data.ECCENTRICITY), title("Rice Eccentricity"); 
figure, histogram(rice_data.ECCENTRICITY), title("Rice Eccentricity");
% Outliers-15 values
rice_data(isoutlier(rice_data.ECCENTRICITY),:);

% --- MajorAXIS-Data exploration ---
figure, boxplot(rice_data.MAJORAXIS), title("Rice Major Axis"); 
figure, histogram(rice_data.MAJORAXIS), title("Rice Major Axis");
% Outliers-0 values
rice_data(isoutlier(rice_data.MAJORAXIS),:);

% --- Perimeter-Data exploration ---
figure, boxplot(rice_data.PERIMETER), title("Rice Perimeter"); 
figure, histogram(rice_data.PERIMETER), title("Rice Perimeter");
% Outliers-0 values
rice_data(isoutlier(rice_data.PERIMETER),:);

% --- Outliers ---
% 45 outliers found-It represents 1% of the sample(45/3810)
outliers = rice_data(any(isoutlier(rice_data(:,1:7)),2),:);
fprintf("Rice Data Outliers: %i\n\n",height(outliers))
% Outliers class distribution is unbalanced, but due to the low percentage,
% we consider as "save" to remove these values from the main sample.
figure, histogram(outliers(:,8).CLASS), title("Rice Outliers");
rice_data=rice_data(~any(isoutlier(rice_data(:,1:7)),2),:);

% --- Response exploration ---
% There is not "significant" class unbalance to be worried about.
figure, histogram(rice_data.CLASS), title("Rice Classes");

% --- Normalizing data ---
% Due to the central limit theorem we assume normality on the data
% so we normalize all variables to a normal standard ~N(0,1) distribution
col_means = mean(table2array(rice_data(:, 1:7)));
col_sd = std(table2array(rice_data(:, 1:7)));
rice_data(:, 1:7) = array2table((table2array(rice_data(:, 1:7)) - ...
    col_means) ./ col_sd);

% --- Correlations ---
fprintf("Rice Data Correlation Matrix")
corrcoef(table2array(rice_data(:,1:7)))

% Correlation of more than .9 beetwen variables: 
% AREA, PERIMETER, MAJORAXIS AND CONVEX_AREA
fprintf("Rice Data High Correlation Features (>0.9)")
corrcoef(table2array(rice_data(:,[1,2,3,6])))

% We use PCA to reduce the dimension of the initial dataset
[coeff,score,latent] = pca(table2array(rice_data(:,[1,2,3,6])));
fprintf("Rice Data (AREA, PERIMETER, MAJORAXIS AND CONVEX_AREA) PCA Explained Variance")
display(latent)

% In this case with just 1 principal component we can explain most of the 
% variability of the model.
rice_data = removevars(rice_data,{'AREA', ...
    'PERIMETER','MAJORAXIS','CONVEX_AREA'});
rice_data(:,"PC1-IMG-FORM")=array2table(score(:,1));

% Re-arrange dataset for simplicity
rice_data = rice_data(:,{'PC1-IMG-FORM','MINORAXIS', ...
    'ECCENTRICITY','EXTENT','CLASS'});

% Save the rice data preprocessed
% Shuffle the rows to avoid inner sampling problem when training the models
rice_data = rice_data(randperm(size(rice_data, 1)), :);
save('data/rice_data.mat','rice_data');



% ++++++++++++++++++++++++++++++++++++++++++++
% ++++++ Concrete Dataset Preprocessing ++++++
% ++++++++++++++++++++++++++++++++++++++++++++
% Data size (3810 x 8)
size(concrete_data);

% Missing values-0 values missing
concrete_data(any(ismissing(concrete_data),2),:);
concrete_data.Properties.VariableNames;
head(concrete_data);

% --- Cement_component1__kgInAM_3Mixture_-Data exploration ---
figure, boxplot(concrete_data.Cement_component1__kgInAM_3Mixture_), title("Concrete Component 1");
figure, histogram(concrete_data.Cement_component1__kgInAM_3Mixture_), title("Concrete Component 1");
% Outliers-0 values
concrete_data(isoutlier(concrete_data.Cement_component1__kgInAM_3Mixture_),:);

% --- BlastFurnaceSlag_component2__kgInAM_3Mixture_-Data exploration ---
figure, boxplot(concrete_data.BlastFurnaceSlag_component2__kgInAM_3Mixture_), title("Concrete Component 2");
figure, histogram(concrete_data.BlastFurnaceSlag_component2__kgInAM_3Mixture_), title("Concrete Component 2");
% Outliers-324 values
concrete_data(isoutlier(concrete_data.BlastFurnaceSlag_component2__kgInAM_3Mixture_),:);

% --- FlyAsh_component3__kgInAM_3Mixture_-Data exploration ---
figure, boxplot(concrete_data.FlyAsh_component3__kgInAM_3Mixture_), title("Concrete Component 3");
figure, histogram(concrete_data.FlyAsh_component3__kgInAM_3Mixture_), title("Concrete Component 3");
% Outliers-46 values
concrete_data(isoutlier(concrete_data.FlyAsh_component3__kgInAM_3Mixture_),:);

% --- Water_component4__kgInAM_3Mixture_-Data exploration ---
figure, boxplot(concrete_data.Water_component4__kgInAM_3Mixture_), title("Concrete Component 4");
figure, histogram(concrete_data.Water_component4__kgInAM_3Mixture_), title("Concrete Component 4");
% Outliers-13 values
concrete_data(isoutlier(concrete_data.Water_component4__kgInAM_3Mixture_),:);

% --- Superplasticizer_component5__kgInAM_3Mixture_-Data exploration ---
figure, boxplot(concrete_data.Superplasticizer_component5__kgInAM_3Mixture_), title("Concrete Component 5");
figure, histogram(concrete_data.Superplasticizer_component5__kgInAM_3Mixture_), title("Concrete Component 5");
% Outliers-5 values
concrete_data(isoutlier(concrete_data.Superplasticizer_component5__kgInAM_3Mixture_),:);

% --- CoarseAggregate_component6__kgInAM_3Mixture_-Data exploration ---
figure, boxplot(concrete_data.CoarseAggregate_component6__kgInAM_3Mixture_), title("Concrete Component 6");
figure, histogram(concrete_data.CoarseAggregate_component6__kgInAM_3Mixture_), title("Concrete Component 6");
% Outliers-0 values
concrete_data(isoutlier(concrete_data.CoarseAggregate_component6__kgInAM_3Mixture_),:);

% --- FineAggregate_component7__kgInAM_3Mixture_-Data exploration ---
figure, boxplot(concrete_data.FineAggregate_component7__kgInAM_3Mixture_), title("Concrete Component 7");
figure, histogram(concrete_data.FineAggregate_component7__kgInAM_3Mixture_), title("Concrete Component 7");
% Outliers-5 values
concrete_data(isoutlier(concrete_data.FineAggregate_component7__kgInAM_3Mixture_),:);

% --- Age_day_-Data exploration ---
figure, boxplot(concrete_data.Age_day_), title("Concrete Age");
figure, histogram(concrete_data.Age_day_), title("Concrete Age");
% Outliers-59 values
concrete_data(isoutlier(concrete_data.Age_day_),:);

% --- Outliers ---
% 772 outliers based on our actual definition-It represents 74% of the data
outliers = concrete_data(any(isoutlier(concrete_data(:,1:8)),2),:);

% Its obvious that we cannot remove 74% of the data.
% The amount of outliers can be explain due to the underlaying process 
% which produces the data. It seems than the data may not come from a
% normal distribution. Thankfully, normality is not an assumption
% of SVM so we can continue without transforming the data.

% Counterpoint: The data could come from a normal distribution with a wide SD

% --- Response exploration ---
% There is no "significant" class unbalance to be worried about.
figure, histogram(concrete_data.ConcreteCompressiveStrength_MPa_Megapascals_), title("Concrete Compressive Strength");
figure, boxplot(concrete_data.ConcreteCompressiveStrength_MPa_Megapascals_), title("Concrete Compressive Strength");
concrete_data(isoutlier(concrete_data.ConcreteCompressiveStrength_MPa_Megapascals_),:);

% --- Normalizing data ---
% Due to the findings in the distribuition of the data, a Z-score
% normalization can be inadecuade so we will use a min-max
% normalization but using percentiles instead of absolutes min and max
% values
col_min = prctile(table2array(concrete_data(:, 1:8)),5);
col_max = prctile(table2array(concrete_data(:, 1:8)),95);
concrete_data(:, 1:8) = array2table((table2array(concrete_data(:, 1:8)) ...
    - col_min)./(col_max-col_min));

% We just want to reduce the difference effect beetween variables so for 
% now we won't be doing any floor or ceiling clipping to 1 o 0.

% --- Correlations ---
% No significant correlations found
fprintf("Concrete Data Correlation Matrix")
corrcoef(table2array(concrete_data(:,1:8)))

% Shuffle the rows to avoid inner sampling problem when training the models
concrete_data = concrete_data(randperm(size(concrete_data, 1)), :);
save('data/concrete_data.mat','concrete_data');
