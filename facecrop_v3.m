%This program detect the faces on the image using Viola-Jones face detection algorithm 
%then crop them and write with new size and name 

clear all;
close all;
clc;

ems={'angry','disgust','fear','happy','neutral','sad','surprise'}
% em={'an','di','fe','ha','ne','sa','su'}

for j=1:7
currentpath(j) =fullfile('C:\ridvan_calismalar\bap_fer\BSEU_FER2', ems(j),'*.jpg');
  
imagefiles = dir(currentpath{j});     
nfiles = length(imagefiles);  

%calling Viola-Jones face detection algorithm
faceDetector = vision.CascadeObjectDetector;

%creating loop for detecting all faces 
for ii=1:nfiles
   currentfilename =fullfile(imagefiles(ii).folder, imagefiles(ii).name);
   currentimage = imread(currentfilename);
   images{ii} = currentimage;
   
   picture = images{ii};
   bboxes = step(faceDetector, picture);
   [m,n] = size(bboxes);
   for i=1:1:m
      
        I2 = imcrop(picture,bboxes(i,:));
        I2 = imresize(I2,[227,227]);
        currentpath2 =fullfile('C:\ridvan_calismalar\bap_fer\BSEU_FER2_Crop', ems(j));
        newimagename = fullfile(currentpath2, imagefiles(ii).name);
        imwrite(I2,newimagename{1});
   end 
   
end

end