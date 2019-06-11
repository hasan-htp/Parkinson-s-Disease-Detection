% bagged tree 
DataSet = readtable('DataSet.xlsx');
fold=5;

inputTable = DataSet;
predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;

% Train a classifier
classificationEnsemble = fitensemble(...
    predictors, ...
    response, ...
    'Bag', ...
    30, ...
    'Tree', ...
    'Type', 'Classification', ...
    'ClassNames', [0; 1]);

% predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));


trainedClassifier.RequiredVariables = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% predictors and response
inputTable = DataSet;
predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;

% Kfolds cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', fold);
fprintf('Fold=%i',fold);
fprintf('\n');

% validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
fprintf('dogruluk orani = %8.8f',validationAccuracy);
fprintf('\n');
% Compute validation predictions and scores
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

