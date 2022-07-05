# Lab report 1
THe lab report can be found [here](https://www.overleaf.com/project/61783465e2867508de9f26a7)

## Project structure
### preprocessing.m
Preprocessing of the both datasets used in this project. Preprocess operations:
1. Data exploration.
2. Outliers treatment.
3. Normalization.
4. Correlation exploration.
5. Dimension reduction (in one of the datasets).

### task_1.m
Lab task 1. Use as the classifier a linear SVM, with the box-constraint C (which determines the
importance of the slack-variables) set to be always one (i.e. no hyperparameters to
tune)

### task_2.m
Use Matlab’s built-in fitcsvm/fitrsvm to train models, now using RBF and Polynomial
kernels. Write an inner-fold cross-validation routine for finding the optimal
hyperparameters. For classification, these are C and the kernel parameters: 'KernelScale',
sigma for RBF and 'PolynomialOrder', q for the polynomial kernel. For regression, you
have to set 'Epsilon' in addition. 

Do not use the automatic parameter optimisation methods. You must write your own, using only basic functions (although you can
compare against that if you wish to check if you’re doing the right thing). You may
want to do these tests on a subset of the data if training is too slow.
Report for each model you trained how many support vectors were selected, both in
absolute terms and in terms of a % of the training data available.

### task_3.m
Perform 10-fold cross-validation on the binary classification and regression tasks.
Report results for linear, Gaussian, and polynomial kernels. Optimise hyper-parameters
with the inner-cross-validation procedure developed in Task 2.

Report on the classification rate for the classification problems and RMSE for the
regression problems. You may once again opt to use less data (if your selected data is
too large) for your evaluation.

## Datasets
We will be using these datasets:
1. https://archive.ics.uci.edu/ml/datasets/Rice+%28Cammeo+and+Osmancik%29
2. https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

## How to use this project?
1. Clone this repo locally
```bash
git clone git@github.com:UoN-Machine-Learning/lab-1.git
```
2. Open the project folder in Matlab.
3. Enjoy!
4. If you are not a Linux/Mac user, you can use whichever git manager you want.

## How to collaborate in this project?
Disclaimer: We don't allow code pushes directly to main, so each change must be in its own branch. Because we are poor, we cannot block pushes to main using Github premium account, so this will be only a gentleman's agreement only.
1. Create a new branch. 
2. Push your branch.
3. Make a pull request.
4. Ask a team-member to review the changes.
5. Merge your changes.
Side note: All collaborators must be in [Student Collaborators](https://github.com/orgs/UoN-Machine-Learning/teams/students-collaborators) team.
