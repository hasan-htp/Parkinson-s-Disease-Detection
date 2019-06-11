%*** 24/12/2018*********************************************%
%*** ALHASAN ALKHATIB B140100255****************************%
%*** Ses kayitlari ile Parkinson hastaligi tespiti**********%
%*** KnnClassifier.m dosyasi********************************%
%***********************************************************%

% Weighted K-NN
function [konfizyon,validationAccuracy]=KnnClassifier(DataSet,fold)
inputTable = DataSet;

predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;

% Train a classifier
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 11, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);

% predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.RequiredVariables = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% predictors and response

inputTable = DataSet;
predictorNames = {'Jitta', 'jitt', 'jit_rap', 'jit_ppq5', 'jit_DDP', 'sh_DB', 'shimmer', 'sh_apq3', 'sh_apq5', 'sh_apq11', 'shim_DDP', 'median_pitch', 'mean_pitch', 'max_pitch', 'min_pitch', 'range_pitch', 'variation', 'oto_KT', 'oto_K0', 'HNR', 'NHR'};
predictors = inputTable(:, predictorNames);
response = inputTable.class;

% cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', fold);


% validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');


% validation predictions and scores
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
