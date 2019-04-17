%Script to extract SIFT features from images (which must all be in
%the same folder)
%
%This script relies on the vlfeat library, available on 
%http://www.vlfeat.org/install-matlab.html.
%
%Authors: Vincent Bazinet

%%
%SOME CONSTANT PARAMETERS
run('vlfeat-0.9.21/toolbox/vl_setup'); %Setup the vlfeat library
path = 'my_new_folder_2'
myFolderInfo = dir(path); %Must specify folder
%%
%FEATURE EXTRACTOR

descriptors1 = [];
descriptors2 = [];
descriptors3 = [];
descriptors4 = [];

%For each images, get SIFT descriptors, at each resolution
for i=3:numel(myFolderInfo)
    ii=i
    imgInfo = myFolderInfo(i);
    img = imgInfo.name;
    path = join([path,'/',img]);
    img = imread(path);
    I = single(rgb2gray(img));
    for k=1:4
        [f,d] = vl_sift(I);
        if k==1
            descriptors1 = [descriptors1, d];
        end
        if k==2
            descriptors2 = [descriptors2, d];
        end
        if k==3
            descriptors3 = [descriptors3, d];
        end
        if k==4
            descriptors4 = [descriptors4, d];
        end
        I = imresize(I,0.50);
    end
end
%%
%COMPUTE CENTROIDS OF CLUSTERS

[centers1_200, ~] = vl_kmeans(single(descriptors1), 500,'Verbose','NumRepetitions',3);
[centers2_200, ~] = vl_kmeans(single(descriptors2), 500,'Verbose','NumRepetitions',3);
[centers3_200, ~] = vl_kmeans(single(descriptors3), 500,'Verbose','NumRepetitions',3);
[centers4_200, ~] = vl_kmeans(single(descriptors4), 500,'Verbose','NumRepetitions',3);

%%
%CREATE FEATURES

features500 = zeros(numel(myFolderInfo)-2,501);

for i=3:numel(myFolderInfo)
    ii=i
    
    imgInfo = myFolderInfo(i);
    img = imgInfo.name;
    path = join([path,'/',img]);
    img = imread(path);
    I = single(rgb2gray(img)); 
    
    kdtree1_500 = vl_kdtreebuild(single(centers1_500));
    
    [f,d] = vl_sift(I);
    for m=1:size(d,2)
        [index500, ~] = vl_kdtreequery(kdtree1_500, single(centers1_500), single(d(:,m)));
        features500(i-2,index500) = features500(i-2,index500)+1;
    end
    
    features500(i-2,501) = nnz(features500(i-2,:));
    
end
