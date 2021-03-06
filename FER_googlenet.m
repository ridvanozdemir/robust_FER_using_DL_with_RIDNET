%This program is for transfer learning using GoogLeNet pre-trained weights

%% Facial Emotion Recognition with GoogLeNet

%Loading training dataset 
allImages = imageDatastore('C:\edmem\seminer2018\RidNet_C', 'IncludeSubfolders', true,'LabelSource', 'foldernames');
[trainingImages, validationImages] = splitEachLabel(allImages, 0.8, 'randomize'); 
%trainingImages = imageDatastore('crop_fer_dataset_1800_train', 'IncludeSubfolders', true,'LabelSource', 'foldernames');

trainingImages.countEachLabel
%Loading test dataset 
testImages = imageDatastore('C:\edmem\deep l\Facial Emotion Recognition 227\crop_CE_fer_dataset_100_test', 'IncludeSubfolders', true,'LabelSource', 'foldernames');

testImages.countEachLabel
validationImages.countEachLabel
%% Pre-trained Network (GoogLeNet) 
net = googlenet;

% Extract the layer graph from the trained network and plot the layer graph.
lgraph = layerGraph(net);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)

% The first element of the Layers property of the network is the image input layer. 
% This layer requires input images of size 224-by-224-by-3, where 3 is the number of color channels. 
net.Layers(1)
inputSize = net.Layers(1).InputSize;

%% Replace Final Layers

lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
numClasses = numel(categories(trainingImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
%% Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);
%% resize the images
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationImages);

% hyperparameter tuning
options = trainingOptions('sgdm', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% Train the network
net = trainNetwork(augimdsTrain,lgraph,options);

%% Classify Validation Images

%Classify the validation images using the fine-tuned network, and calculate the classification accuracy.
[YPred,probs] = classify(net,augimdsValidation);
validation_accuracy = mean(YPred == validationImages.Labels)

%% Test Network Performance

testImages.ReadFcn = @readFunctionTrain_224; 
predictedLabels = classify(net, testImages); 
test_data_accuracy = mean(predictedLabels == testImages.Labels)

% confusion matrix - heat map

confMat = confusionmat(testImages.Labels, predictedLabels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))
tt = table(testImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
figure; heatmap(tt,'Predicted','Actual');