%% MAE156 Phone image processing MATLAB test code
% Creates mobile connection to phone via chord and package
% Setup: Have white background within an enclosed area. 
%   - Position phone camera about 1/3 foot abover the specimen. 
%   - Orient the specimen to have the handles be more 
%       directly under the flash light of the camera. 
%   - (Test setup) Have the clamps near the bottom left corner
clearvars;

%% Creation of connection with IOS phone
clearvars; % Needs to clear previous camera

m = mobiledev; % Connection with phone
cam = camera(m,'back');

% Camera properties
cam.Resolution = '1280x720';
%cam.Resolution = '720x1280';
cam.Autofocus = 'on';
%cam.Flash = 'off';
%cam.Autofocus = 'on';
cam.Flash = 'on';

%% Aquiring snapshot

% img = snapshot(cam,'manual');
img_test = snapshot(cam,'immediate');
%daspect([1,1,1])
%image(img_test)

%% Image processing

%Landscape (Bottom middle)
%Crop_im=imcrop(img_test,[730,300,200,100]);

%Portrait (Experimental)
Crop_im=imcrop(img_test,[300,750,200,100]);

Gray = rgb2gray(Crop_im);

%Gray = rgb2gray(img_test);
%Gray=imcrop(Gray,[730,300,200,100]);
%BW = imbinarize(Fil, 'adaptive', 'ForegroundPolarity','bright','Sensitivity',0.8);

%Experimental
%Gray=wiener2(Gray,[4,4],20);


% Feature Detection

edge_d='Canny';
%Landscape
% min_th=0.05;
% max_th=0.32;

%Portrait
min_th=0.01;
max_th=0.231;

thre=[min_th max_th];
Gray2=edge_cl(Gray,edge_d,thre);
%BW2=edge(BW,edge_d,[min_th,max_th]);

% Rotate image if tips are oriented down
[t_centers, t_radii, t_metric] = imfindcircles(Gray2,[6 16]);
if ceil(mean(t_centers(:,2))) > size(Gray2,1)*0.5
    Gray2 = imrotate(Gray2,180);
    Crop_im = imrotate(Crop_im,180);
end



[g_centers, g_radii, g_metric] = imfindcircles(Gray2,[6 16]);
%[centers, radii, metric] = imfindcircles(BW2,[6 16]);

figure(1);
hold on
imshow(Gray2)
imshowpair(Gray2,Gray,'montage')
%viscircles(centers, radii,'Color','b');
viscircles(g_centers, g_radii,'Color','b');

%imshowpair(Gray2,BW2,'montage')
%imshow(BW2)
%viscircles(centers, radii,'Color','b');
hold off
truesize(1, [300,300]);

%% Plotting lines for the clamp jaws
% Makes a line of the edge of the clamp jaws by finding the end points in
% contact with the edge and of that not passing the circle center y value

% End point values for the circle centers
start_y = ceil(0.75*size(Gray2,1));
end_y = ceil(min(g_centers(1:end,2))) + ceil(0.05*size(Gray2,1));

% Initiate vector with line's corresponding start and ends points
jaw_x1 = zeros(size(Gray2,2));
jaw_x2 = zeros(size(Gray2,2));
prev_x1 = 1;
prev_x2 = 1;
point_tol = 2;

% Finding the start and end points
for ii = 1:size(Gray2,2)
    if Gray2(start_y,ii) == 1 && abs(ii - prev_x1) >= point_tol
        jaw_x1(ii) = ii;
        prev_x1 = ii;
    end
    if Gray2(end_y, ii) == 1 && abs(ii - prev_x2) >= point_tol
        jaw_x2(ii) = ii;
        prev_x2 = ii;
    end

end 

jaw_x1 = jaw_x1(jaw_x1~=0)';
jaw_x2 = jaw_x2(jaw_x2~=0)';

% If the vectors not the same size, lines can't be plot
if length(jaw_x1) ~= length(jaw_x2)
    print("Error in image processing. Please retake shot")
end


figure(2);
imshow(Gray2)
hold on
for kk = 1:length(jaw_x1)
    plot([jaw_x1(kk),jaw_x2(kk)], [start_y,end_y], 'LineWidth', 8,...
        'Color', 'g')
end 
yline(start_y, ":", "Offset", "LineWidth", 3, "Color", "yellow")
yline(end_y, ":", "Jaw Tips", "LineWidth", 3, "Color", "yellow")
hold off
truesize(2, [300,200]);

%% Computing Distance between the jaws
% Isolating the indiviaul clamps
% Intiating clamp arrays
clear clamp2_x2 clamp1_x2 clamp2_x1 clamp1_x1;
clamp1_x1 = zeros(1,4);
clamp2_x1 = zeros(1,4);
clamp1_x2 = zeros(1,4);
clamp2_x2 = zeros(1,4);
edge_cl = 0;

% Compare proximity wrt points recorded at jaw tip
for ii = 1:length(jaw_x2)
    if jaw_x2(end) - jaw_x2(ii) >= 0.35*size(Gray2,2) && edge_cl < 3
        clamp1_x2(ii) = jaw_x2(ii);
        clamp1_x1(ii) = jaw_x1(ii);
    else 
        clamp2_x2(ii) = jaw_x2(ii);
        clamp2_x1(ii) = jaw_x1(ii);
    end 
end 
clamp1_x1 = clamp1_x1(clamp1_x1~=0)';
clamp2_x1 = clamp2_x1(clamp2_x1~=0)';
clamp1_x2 = clamp1_x2(clamp1_x2~=0)';
clamp2_x2 = clamp2_x2(clamp2_x2~=0)';

clamp1_y1 = start_y.*ones(1,length(clamp1_x1));
clamp1_y2 = end_y.*ones(1,length(clamp1_x2));
clamp2_y1 = start_y.*ones(1,length(clamp2_x1));
clamp2_y2 = end_y.*ones(1,length(clamp2_x2));

% Finding the angle
% Use the cross product value formula to obtain angles of the lines
% Report the biggest angle
theta1 = zeros(1,2);
theta2 = zeros(1,2);

for kk = 1:(length(clamp1_x1) - 1)
    v_ref = [clamp1_x2(1) - clamp1_x1(1), clamp1_y2(1) - clamp1_y1(1)];
    v_check = [clamp1_x2(kk+1) - clamp1_x1(kk+1), ...
        clamp1_y2(kk+1) - clamp1_y1(kk+1)];
    theta1(kk) = acos(dot(v_ref, v_check)/(norm(v_ref)*norm(v_check)));
end 
for gg = 1:(length(clamp2_x1) - 1)
    v_ref = [clamp2_x2(1) - clamp2_x1(1), clamp2_y2(1) - clamp2_y1(1)];
    v_check = [clamp2_x2(gg+1) - clamp2_x1(gg+1), ...
        clamp2_y2(gg+1) - clamp2_y1(gg+1)];
    theta2(gg) = acos(dot(v_ref, v_check)/(norm(v_ref)*norm(v_check)));
end 


theta1 = theta1(theta1~=0)';
theta2 = theta2(theta2~=0)';

if length(theta1) > 1
    theta1 = max(theta1);
else
    theta1 = 0;
end

if length(theta2) > 1
    theta2 = max(theta2);
else
    theta2 = 0;
end


% Finding the distances between the jaws
% Values in mm
jaw_length = 13;
jaw_width = 1.75;
clamp1_distance = 2*theta1*jaw_length;
clamp2_distance = 2*theta2*jaw_length;

if clamp1_distance >= jaw_width/2
    clamp1label = 'red';
    weight1 = 'bold';
else 
    clamp1label = 'green';
    weight1 = 'normal';
end
if clamp2_distance >= jaw_width/2
    clamp2label = 'red';
    weight2 = 'bold';
else 
    clamp2label = 'green';
    weight2 = 'normal';
end


figure(3);
imshow(Crop_im)
hold on
for kk = 1:length(jaw_x1)
    plot([jaw_x1(kk),jaw_x2(kk)], [start_y,end_y], 'LineWidth', 8,...
        'Color', 'g')
end 
yline(start_y, ":", "Offset", "LineWidth", 3, "Color", "yellow")
yline(end_y, ":", "Jaw Tips", "LineWidth", 3, "Color", "yellow")

text(mean(clamp1_x2), end_y - ceil(0.1*size(Gray2,1)), ...
    "Largest jaw distance: " + clamp1_distance, "Color", clamp1label,...
    "FontSize", 14, "FontWeight", weight1);
text(mean(clamp2_x2), end_y - ceil(0.1*size(Gray2,1)), ...
    "Largest jaw distance: " + clamp2_distance, "Color", clamp2label,...
    "FontSize", 14, "FontWeight", weight2);
title("Processed clamp photo with annotations", "FontSize", 17)

hold off
truesize(3, [300,200]);



