%*** 24/12/2018*********************************************%
%*** ALHASAN ALKHATIB B140100255****************************%
%*** Ses kayitlari ile Parkinson hastaligi tespiti**********%
%*** Ensemble_GentleBoost.m dosyasi*************************%
%***********************************************************%

%Ensemble Algoritmasi
%Karar agaclari
%GentleBoost 



function [konfizyon,validationAccuracy]=Ensemble_GentleBoost(DataSet,fold)

validationAccuracy=0;
inputTable = DataSet;
predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 20);
V=0.92;
classificationEnsemble = fitensemble(...
    predictors, ...
    response, ...
    'GentleBoost', ...
    30, ...
    template, ...
    'Type', 'Classification', ...
    'LearnRate', 0.1, ...
    'ClassNames', [0; 1]);
%while(validationAccuracy<V)

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% predictors and response
inputTable = DataSet;
predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', fold);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
%end
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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
