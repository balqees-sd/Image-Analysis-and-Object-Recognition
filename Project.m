# Final Project - Bag of Words with Template Matching
# ======================= Group 22 =====================

function Project
# Initialize package and clear variables
clear all
close all
clc
pkg load image

% -------------------------------- Task 1 --------------------------------

# Read the image, the template and normilized them
img=imread('./images/1.jpg');
template=imread('./words/1.jpg');

imggray = rgb2gray(img);
templategray=rgb2gray(template);

imgnorm = double(imggray)/255;
templatenorm = double(templategray)/255;

# a-b-c) Template matching

[outTM] = Template_Matching(imgnorm,templatenorm);

# Find the highest Normalized cross-correlation

[hr,wr] = find(outTM == max(abs(outTM(:))));

figure;
subplot(1,3,1)
imshow(img);
title("Image");
subplot(1,3,2)
imshow(template);
title("Template");
subplot(1,3,3)
imshow(img);
title("Template matching");
hold on;
rectangle ("Position", [wr,hr,size(templatenorm,2),size(templatenorm,1)],"EdgeColor","red");

# d) Template matching with different scales

imgreduce=imgnorm;

figure
for(i=1:4)

  if(size(imgreduce,1) > size(templatenorm,1) && size(imgreduce,2) > size(templatenorm,2))

    [outTM] = Template_Matching(imgreduce,templatenorm);
    [hr,wr] = find(outTM == max(abs(outTM(:))));

  endif

  subplot(1,4,i)
  imshow(imgreduce);
  title(cstrcat("Scale : ", num2str(size(imgreduce,1)/size(imgnorm,1),2)," NCC : ",num2str(max(abs(outTM(:))),4)));
  hold on;
  rectangle ("Position", [wr,hr,size(templatenorm,2),size(templatenorm,1)],"EdgeColor","red");
  imgreduce = imresize(imgreduce,0.5);

endfor

# e) Plot four exemplary results

figure;
for(i=1:4)

  if(i == 1)
    img = imread('./images/1.jpg');
    template=imread('./words/1.jpg');
  elseif (i == 2)
    img = imread('./images/2.jpg');
    template=imread('./words/2.jpg');
  elseif (i == 3)
    img = imread('./images/3.jpg');
    template=imread('./words/4.jpg');
  elseif (i == 4)
    img = imread('./images/4.jpg');
    template=imread('./words/5.jpg');
  endif

  imggray = rgb2gray(img);
  templategray=rgb2gray(template);

  imgnorm = double(imggray)/255;
  templatenorm = double(templategray)/255;
  [outTM] = Template_Matching(imgnorm,templatenorm);
  [hr,wr] = find(outTM == max(abs(outTM(:))));

  subplot(2,4,i)
  imshow(img);
  title(cstrcat("Image :  ", int2str(i)));
  hold on;
  rectangle ("Position", [wr,hr,size(templatenorm,2),size(templatenorm,1)],"EdgeColor","red");
  subplot(2,4,i+4);
  imshow(template);
  title(cstrcat("Template :  ", int2str(i)));

endfor

% -------------------------------- Task 2 --------------------------------

%a) Feature Vector for 1 image

img=imread('./images/5.jpg');
imggray = rgb2gray(img);
imgnorm = double(imggray)/255;

resultFV = Feature_Vector(imgnorm);

figure
plot(resultFV,'x');
axis ([0 7 0.4 1]);
xlabel('Visual Word', 'Fontsize',18);
ylabel('Normalized Cross-correlation', 'Fontsize',18);
title ("Feature Vector for image 5");

%b) Feature Vector for all images

BVW=[]; % Array for the bag of visual words

for(k=1:6)

  img=imread(["./images/",int2str(k),".jpg"]);;
  imggray = rgb2gray(img);
  imgnorm = double(imggray)/255;

  resultFV = Feature_Vector(imgnorm);

  BVW(k,:) = resultFV;

endfor

figure
plot(BVW(1,:),'x');
hold on;
plot(BVW(2,:),'+');
hold on;
plot(BVW(3,:),'*');
hold on;
plot(BVW(4,:),'o');
hold on;
plot(BVW(5,:),'s');
hold on;
plot(BVW(6,:),'d');
hold on;
axis ([0 7 0.4 1]);
legend ({"Image 1","Image 2","Image 3","Image 4","Image 5","Image 6"}, "location", "east");
xlabel('Visual Word', 'Fontsize',18);
ylabel('Normalized Cross-correlation', 'Fontsize',18);
title ("Bag of visual worlds");

%c-d) KNN - k = 3
% Use images 1 to 4 as training sample and image 5 as test
votes=KNN(BVW(5,:),BVW(1:4,:));

% Predict the class
if(votes(1) > votes(2))
  predict="Violin";
else
   predict="Guitar";
endif

figure;
img=imread('./images/5.jpg');
imshow(img);
title(cstrcat("Predicted class : ", predict," / Votes   Violin:",num2str(votes(1)),"   Guitar:",num2str(votes(2))));

% Use images 1 to 4 as training sample and image 6 as test
votes=KNN(BVW(6,:),BVW(1:4,:));

% Predict the class
if(votes(1) > votes(2))
  predict="Violin";
else
   predict="Guitar";
endif

figure;
img=imread('./images/6.jpg');
imshow(img);
title(cstrcat("Predicted class : ", predict," / Votes   Violin:",num2str(votes(1)),"   Guitar:",num2str(votes(2))));

endfunction

%k-nearest neighbors
function votes = KNN(imgFeatureVector,testFeatureVector)

   votes=[0,0]; % votes(1) = Violin / votes(2) = Guitar

  [h,w] = size(testFeatureVector);

  % Create and calculate the distance
  distance=[];

  for(i=1:h)
      distance(i) = norm(imgFeatureVector-testFeatureVector(i,:));
  endfor

  % Sort the items and choose the 3 nearest neighbors
  [out,indices] = sort(distance);
  indices=int8(indices);
  for(i=1:3)
    % put one vote for the 3 nearest neighbots
    if(indices(i) == 1 || indices(i) == 2 || indices(i) == 5)
      votes(1) += 1;
    elseif (indices(i) == 3 || indices(i) == 4 || indices(i) == 6)
      votes(2) += 1;
    endif

  endfor

endfunction

function result=Feature_Vector(img)

% Create the loop for evaluate the different visual words
result=[];

for(j=1:6)

imgreduce=img;

template=imread(["./words/",int2str(j),".jpg"]);;
templategray=rgb2gray(template);
templatenorm = double(templategray)/255;

bestGlobalNCC=0; % Best NCC over multiple scales

% Create a loop for evaluate different scales

  for(i=1:4)

    if(size(imgreduce,1) > size(templatenorm,1) && size(imgreduce,2) > size(templatenorm,2))

      [outTM] = Template_Matching(imgreduce,templatenorm);
       bestLocalNCC = max(abs(outTM(:)));
      [hr,wr] = find(outTM == bestLocalNCC);

      if(bestGlobalNCC < bestLocalNCC)
        bestGlobalNCC = bestLocalNCC;
      endif

    endif

    imgreduce = imresize(imgreduce,0.5);

  endfor

  result(j) = bestGlobalNCC;

endfor

endfunction

function [out,hr,wr] = Template_Matching(img,template)

# Match template at current scale with the same with

[h1,w1] = size(img);
[h2,w2] = size(template);

out=zeros(h1,w1);

padi=uint16(floor(h2/2));
padj=uint16(floor(w2/2));

sigmaB=stddev(template); # Calculate sigma of the template

for(i=1:(h1-h2))
  auxi1=i;
  auxi2=i+h2-1;
  for(j=1:(w1-w2))
    auxj1=j;
    auxj2=j+w2-1;
    patch=img(auxi1:auxi2,auxj1:auxj2);
    sigmaA=stddev(patch);
    if (sigmaA != 0 && sigmaB != 0)
      sigmaAB=covariance(patch,template);
      out(i,j)= sigmaAB/(sqrt((sigmaA)*(sigmaB)));
    endif
  endfor
endfor

endfunction

%Calculate standard deviation
function result=stddev(img)

  [h,w] = size(img);

  sumtot=0;

  sumtot = img.*img;
  sumtot = sum(sumtot(:));

  result = ((1/(w^2))*sumtot) - mean(img(:))^2;

endfunction

%Calculate covariance
function result=covariance(img,template)

  [h,w] = size(img);

  sumtot=0;

  sumtot = img.*template;
  sumtot = sum(sumtot(:));

  result = ((1/(w^2))*sumtot) - mean(img(:))*mean(template(:));

endfunction
