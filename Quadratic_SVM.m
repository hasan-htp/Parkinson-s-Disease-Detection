%*** 24/12/2018*********************************************%
%*** ALHASAN ALKHATIB B140100255****************************%
%*** Ses kayitlari ile Parkinson hastaligi tespiti**********%
%*** Main.m dosyasi*****************************************%
%***********************************************************%

function Quadratic_SVM(DataSet,fold)

inputTable = DataSet;

predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
inputTable = DataSet;
predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;

% Perform cross-validation

partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', fold);
fprintf('Fold=%i',fold);
fprintf('\n');

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
fprintf('Quadratic_SVM icin dogruluk orani = %8.8f',validationAccuracy);
fprintf('\n');

% Compute validation predictions and scores
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
end
