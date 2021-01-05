%This program detects the faces on the image using Viola-Jones face detection algorithm 
%then crops and pre processes them and writes with new size and name 
%Ridvan Ozdemir


close all;
clear all;
imagefiles = dir('C:\edmem\seminer2018\AffectNet\angry\*.jpg');     
nfiles = length(imagefiles);    % Number of files found

faceDetector = vision.CascadeObjectDetector;

for ii=1:nfiles
   currentfilename =fullfile('C:\edmem\seminer2018\AffectNet\angry\', imagefiles(ii).name);
   currentimage = imread(currentfilename);
   images{ii} = currentimage;
   
   picture = images{ii};
   bboxes = step(faceDetector, picture);
   [m,n] = size(bboxes);
   for i=1:1:m
      
        I2 = imcrop(picture,bboxes(i,:));
        I2 = imresize(I2,[224,224]);
        folder = 'C:\edmem\seminer2018\AffectNet_C\angry\';
        newimagename = [folder imagefiles(ii).name '_01' '.jpg'];
        imwrite(I2,newimagename);
        
        % pre processing for data augmentation
        x2=imadjust(I2,[.2 .2 .2; .8 .8 .8],[]);
        x3=x2-50; % darker
        x5=x2+60; % brighter
        
        newimagename_d = [folder imagefiles(ii).name '_01_d' '.jpg'];
        imwrite(I2,newimagename_d);
        
        newimagename_b = [folder imagefiles(ii).name '_01_b' '.jpg'];
        imwrite(I2,newimagename_b);
        
        imwrite(x3,newimagename_d);
        imwrite(x5,newimagename_b);
   end 
   
end
