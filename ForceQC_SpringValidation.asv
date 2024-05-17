%% MAE156 Phone image processing: Spring Bending Tests
% Creates mobile connection to phone via chord and package
% Setup: Have white background within an enclosed area. 
%   - Position phone camera about 1/3 foot abover the specimen. 
%   - Orient the spring to show the side cross section
%   - (Test setup) Have the spring near the bottom left corner

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

%% (1) Undeformed values for strips

p2mm = ceil(231/(25.4));            % pixels per mm
ltest = 55;                         % length from cantilever mount (mm)
start_off = 50;                     % Where to start going through lines



reg = out_find(cam, ltest, p2mm);

%%
[top_d, bot_d] = y_diff(reg, start_off, (p2mm*ltest-5));


%% Finding the start and end points

function [top_d, bot_d] = y_diff(Gray2, start_off, tip)
    prev_y1 = 1;
    prev_y2 = 1;
    point_tol = 5;
    start_yf = zeros(1,5);
    end_yf = zeros(1,5);

    for ii = 1:size(reg,1)
        if Gray2(ii,start_off) == 1 && abs(ii - prev_y1) >= point_tol
            start_yf(ii) = ii;
            prev_y1 = ii;
        end
        if Gray2(ii,tip) == 1 && abs(ii - prev_y2) >= point_tol
            end_yf(ii) = ii;
            prev_y2 = ii;
        end    
    end 

    start_y(1) = min(start_yf(start_yf~=0));
    start_y(2) = max(start_yf(start_yf~=0));
    end_y(1) = min(end_yf(end_yf~=0));
    end_y(2) = max(end_yf(end_yf~=0));

    top_d = abs(end_y(1) - start_y(1));
    bot_d = abs(end_y(2) - start_y(2));
end 



%% Aquiring snapshot

function Gray2 = out_find(cam, ltest, p2mm)

    img_test = snapshot(cam,'immediate');
    image(img_test);
    daspect([1,1,1])
    
    % Obtaining the contour lines
    % Rotate image
    Crop_im=imrotate(img_test,270);
    
    % Crop locations
    x_off = 20;
    x_s = 490;
    x_f = (ltest*p2mm) + x_off;
    
    Crop_im=imcrop(Crop_im,[x_s,100,x_f,200]);
    % image(Crop_im);
    % daspect([1,1,1])
    
    Gray = rgb2gray(Crop_im);
    Gray=wiener2(Gray,[5,5],20);
    
    % Feature Detection
    edge_d='Canny';
    min_th=0.1;
    max_th=0.35;
    
    thre=[min_th max_th];
    Gray2=edge(Gray,edge_d,thre);
    
    % figure(1);
    % hold on
    % imshow(Gray2)
    % %imshowpair(Gray2,Gray,'montage')
    % hold off
    % truesize(1, [500,500]);

end

