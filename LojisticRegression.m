%*** 24/12/2018*********************************************%
%*** ALHASAN ALKHATIB B140100255****************************%
%*** Ses kayitlari ile Parkinson hastaligi tespiti**********%
%*** Ensemble_GentleBoost.m dosyasi*************************%
%***********************************************************%

%Lojistic Regression
function [konfizyon,validationAccuracy]=LojisticRegression(DataSet,KFolds)

inputTable = DataSet;
predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% 1 or true = 'successful' class
% 0 or false = 'failure' class
% NaN - missing response.
successClass = double(1);
failureClass = double(0);
missingClass = double(NaN);
successFailureAndMissingClasses = [successClass; failureClass; missingClass];
isMissing = isnan(response);
zeroOneResponse = double(ismember(response, successClass));
zeroOneResponse(isMissing) = NaN;
% Prepare input arguments to fitglm.
categoricalPredictorIndex = find(isCategoricalPredictor);
concatenatedPredictorsAndResponse = [predictors, table(zeroOneResponse)];

% Train using zero-one responses
% categorical.
GeneralizedLinearModel = fitglm(...
    concatenatedPredictorsAndResponse, ...
    'Distribution', 'binomial', ...
    'link', 'logit', ...
    'CategoricalVars', categoricalPredictorIndex);

% Convert predicted probabilities to predicted class labels and scores.
convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
scoresFcn = @(p) [1-p, p];
predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
trainedClassifier.predictFcn = @(x) logisticRegressionPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
trainedClassifier.GeneralizedLinearModel = GeneralizedLinearModel;
trainedClassifier.SuccessClass = successClass;
trainedClassifier.FailureClass = failureClass;
trainedClassifier.MissingClass = missingClass;
trainedClassifier.ClassNames = {successClass; failureClass};
trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
inputTable = DataSet;
predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation


cvp = cvpartition(response, 'KFold', KFolds);
% Initialize the predictions and scores to the proper sizes
validationPredictions = response;
numObservations = size(predictors, 1);
numClasses = 2;
validationScores = NaN(numObservations, numClasses);

for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;
    
    % Train a classifier
    % 1 or true = 'successful' class
    % 0 or false = 'failure' class
    % NaN - missing response.
    successClass = double(1);
    failureClass = double(0);
    missingClass = double(NaN);
    successFailureAndMissingClasses = [successClass; failureClass; missingClass];
    isMissing = isnan(trainingResponse);
    zeroOneResponse = double(ismember(trainingResponse, successClass));
    zeroOneResponse(isMissing) = NaN;
    % Prepare input arguments to fitglm.
    categoricalPredictorIndex = find(foldIsCategoricalPredictor);
    concatenatedPredictorsAndResponse = [trainingPredictors, table(zeroOneResponse)];
    
    % Train using zero-one responses, specifying which predictors are
    % categorical.
    GeneralizedLinearModel = fitglm(...
        concatenatedPredictorsAndResponse, ...
        'Distribution', 'binomial', ...
        'link', 'logit', ...
        'CategoricalVars', categoricalPredictorIndex);
    
    % Convert predicted probabilities to predicted class labels and scores.
    convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
    returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
    scoresFcn = @(p) [1-p, p];
    predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );
    
    % Create the result struct with predict function
    logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
    validationPredictFcn = @(x) logisticRegressionPredictFcn(x);
    
    % Add additional fields to the result struct
    
    % Compute validation predictions and scores
    validationPredictors = predictors(cvp.test(fold), :);
    [foldPredictions, foldScores] = validationPredictFcn(validationPredictors);
    
    % Store predictions and scores in the original order
    validationPredictions(cvp.test(fold), :) = foldPredictions;
    validationScores(cvp.test(fold), :) = foldScores;
end

correctPredictions = (validationPredictions == response);
validationAccuracy = sum(correctPredictions)/length(correctPredictions);

[N,F]=size(DataSet);

R=response;
VP=validationPredictions;
AA=0;AN=0;NA=0;NN=0;

for i=1:N
   if R(i)==1 &&  VP(i)==1
       AA=AA+1;
   elseif R(i)==1 &&  VP(i)==0
       AN=AN+1;
   elseif R(i)==0 &&  VP(i)==1
       NA=NA+1;
   elseif R(i)==0 &&  VP(i)==0
       NN=NN+1;
   end 
end
konfizyon=zeros(2,2);
konfizyon(1,1)=AA/168;konfizyon(1,2)=AN/168;konfizyon(2,1)=NA/174;konfizyon(2,2)=NN/174;




end
