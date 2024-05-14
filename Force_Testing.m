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
img_test2 = snapshot(cam,'immediate');
%daspect([1,1,1])
figure(1);
image(img_test2)
