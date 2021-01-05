%This program is for creating training dataset from AffectNet's dataset's
%images.
%Ridvan Ozdemir

clear all; 
close all;

%Read csv file to get emotion and path information 
A=importdata('C:\edmem\seminer2018\dataset 2\training.csv');
Emotions=A.data(:,1);
Paths=A.textdata(:,1);
Paths_E=Paths(2:end);


%Get 1000 images for every emotion class and copy them to new direction
for ii=1:1000 

     currentfilename =fullfile('C:\edmem\seminer2018\dataset 2\Manually_Annotated\Manually_Annotated_Images\', Paths_E{ii});
     currentfile{ii}=currentfilename;
     c_file=currentfile{ii};
     
     n = Emotions(ii);
     
    switch n
        case 0
            copyfile(c_file,'C:\edmem\seminer2018\AffectNet\neutral');
        case 1
            copyfile(c_file,'C:\edmem\seminer2018\AffectNet\happy');
        case 2
            copyfile(c_file,'C:\edmem\seminer2018\AffectNet\sad');
        case 3
            copyfile(c_file,'C:\edmem\seminer2018\AffectNet\surprise');
        case 4
            copyfile(c_file,'C:\edmem\seminer2018\AffectNet\fear');
        case 5
            copyfile(c_file,'C:\edmem\seminer2018\AffectNet\disgust');
        case 6
            copyfile(c_file,'C:\edmem\seminer2018\AffectNet\angry');
        otherwise
        
    end

end

