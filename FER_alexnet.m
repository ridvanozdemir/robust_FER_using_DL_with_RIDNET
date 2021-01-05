%This program is for transfer learning using AlexNet pre-trained weights

%Loading training dataset 
allImages = imageDatastore('C:\edmem\seminer2018\RidNet_C', 'IncludeSubfolders', true,'LabelSource', 'foldernames');
[trainingImages, validationImages] = splitEachLabel(allImages, 0.8, 'randomize'); 
%trainingImages = imageDatastore('crop_fer_dataset_1800_train', 'IncludeSubfolders', true,'LabelSource', 'foldernames');

trainingImages.countEachLabel
%Loading test dataset 
%testImages = imageDatastore('C:\edmem\deep l\Facial Emotion Recognition 227\crop_CE_fer_dataset_100_test', 'IncludeSubfolders', true,'LabelSource', 'foldernames');
testImages = imageDatastore('C:\edmem\seminer2018\SFEW_2_C_R', 'IncludeSubfolders', true,'LabelSource', 'foldernames');

testImages.countEachLabel
validationImages.countEachLabel
%% Pre-trained Network (AlexNet) 
alex = alexnet;
%% Show layers 
layers = alex.Layers
layers
%% Set the class number

numClasses = numel(categories(trainingImages.Labels));
 
layers(23) = fullyConnectedLayer(numClasses);  
layers(25) = classificationLayer

%% hyperparameter tuning

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationImages, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% resize and read training data

trainingImages.ReadFcn = @readFunctionTrain;
%% train the network

myNet = trainNetwork(trainingImages, layers, options);
%% Test Network Performance

testImages.ReadFcn = @readFunctionTrain; 
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)

% confusion matrix - heat map 

confMat = confusionmat(testImages.Labels, predictedLabels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))
tt = table(testImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
figure; heatmap(tt,'Predicted','Actual');