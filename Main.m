%*** 24/12/2018*********************************************%
%*** ALHASAN ALKHATIB B140100255****************************%
%*** Ses kayitlari ile Parkinson hastaligi tespiti**********%
%*** Main.m dosyasi*****************************************%
%***********************************************************%

DataSet = readtable('DataSet.xlsx');

%DataSet bir Tablo nesnesidir 
%iknci parametre fold parametresi
fold=342;
N=5;
D_orani=zeros(N,3);
k1=zeros(2,2,N);
k2=zeros(2,2,N);
k3=zeros(2,2,N);

 for i=1:N
 [k1(:,:,i),D_orani(i,1)]=Ensemble_GentleBoost(DataSet,fold);
 [k2(:,:,i),D_orani(i,2)]=KnnClassifier(DataSet,fold);
 [k3(:,:,i),D_orani(i,3)]=LojisticRegression(DataSet,fold);
 end
Dogruluk_orani=sum(D_orani,1)/N;

k1=mean(k1,3);k2=mean(k2,3);k3=mean(k3,3);


fprintf('Ensemble_GentleBoost icin= %8.8f ',Dogruluk_orani(1)*100)
fprintf('\n');
fprintf('Weighted K-NN= %8.8f %',Dogruluk_orani(2)*100)
fprintf('\n');
fprintf('Lojistic Regression icin= %8.8f ',Dogruluk_orani(3)*100)
fprintf('\n');



