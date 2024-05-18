%% MAE156 Phone image processing: Spring Bending Tests
% Creates mobile connection to phone via chord and package
% Setup: Have white background within an enclosed area. 
%   - Position phone camera about 1/3 foot abover the specimen. 
%   - Orient the spring to show the side cross section
%   - (Test setup) Have the spring near the bottom left corner

%% (0) Connection with IOS phone
clear m cam;                        % Needs to clear previous camera

m = mobiledev;                      % Connection with phone
cam = camera(m,'back');

% Camera properties
cam.Resolution = '1280x720';
cam.Autofocus = 'on';
cam.Flash = 'on';

%% (1) Undeformed values for strips

p2mm = ceil(231/(25.4));            % pixels per mm
ltest = 55;                         % length from cantilever mount (mm)
start_off = 20;                     % Where to start going through lines

% Pic of un-deflected strips
reg = out_find(cam, ltest, p2mm);

% Getting differences in height for the top and bottom strips
tip_ap = (p2mm*ltest);
tip_off = 15;

tip = tip_search(reg, tip_ap, 'Undeformed');
[start_y, end_y, top_d, bot_d] = y_diff(reg, start_off, (tip-tip_off));


%% (2) Deformed values for strips

% Pic of deflected strips
def = out_find(cam, ltest, p2mm);
%imshow(def)

% Getting differences in height for the top and bottom strips
%dtip = tip_search(def, tip_ap, 'Deformed');
[dstart_y, dend_y, dtop_d, dbot_d] = y_diff(def, start_off, (tip-tip_off));


%% (3) Force calculation for the clamp

E_mod = 200*10^9;                         % Stainless steel elastic modulus
I_strip = (1/12)*(3.8*0.001)*(0.00064^3); % Strip Cross section inertia
L_strip = ltest*0.001;                    % Strip length (to meters)
N2gf = 101.972;                           % Newtons to gram-force

def_topp = abs(dtop_d - top_d);           % Deflection of top (pixels)
def_botp = abs(dbot_d - bot_d);           % Deflection of bottom (pixels)

def_top = def_topp*(1/p2mm)*0.001;            % Deflection of top (pixels)
def_bot = def_botp*(1/p2mm)*0.001;            % Deflection of bottom (pixels)

% Force calculations assuming cantilever connection
P_top = N2gf*((def_top*3*E_mod*I_strip)/((L_strip)^3));
P_bot = N2gf*((def_bot*3*E_mod*I_strip)/((L_strip)^3));


% Plotting points for verification
figure(1);

subplot(2,1,1)
imshow(reg);
hold on
title('Undeformed Strips')
scatter([tip,tip], end_y, 'LineWidth', 1, 'MarkerFaceColor', 'flat')
scatter([start_off,start_off], start_y, 'LineWidth', ...
    1, 'MarkerFaceColor', 'flat')
xline(start_off, ":", "Offset start", ...
    "LineWidth", 1, "Color", "yellow")
xline(tip, ":", "Jaw Tips", "LineWidth", 1, "Color", "yellow")
hold off

subplot(2,1,2)
imshow(def);
hold on
title('Clamped Strips')
scatter([tip,tip], dend_y, 'LineWidth', 1, 'MarkerFaceColor', 'flat')
scatter([start_off,start_off], dstart_y, 'LineWidth', ...
    1, 'MarkerFaceColor', 'flat')
xline(start_off, ":", "Offset start", ...
    "LineWidth", 1, "Color", "yellow")
xline(tip, ":", "Jaw Tips", "LineWidth", 1, "Color", "yellow")
text(tip - 250, max(dend_y) + 20, "Bottom Clamp Force (gf): " + P_bot, ...
    "Color", "g", "FontSize", 14, "FontWeight", 'bold');
text(tip - 250, min(dend_y) - 20, "Top Clamp Force (gf): " + P_top, ...
    "Color", "g", "FontSize", 14, "FontWeight", 'bold');
hold off


%% Function: Finding Tip

function [tip] = tip_search(Gray2, tip_ap, state)
    
    switch state
        case 'Undeformed'
            t_pt_th = 4;
            start_scan = size(Gray2,2);
        case 'Deformed'
            t_pt_th = 8;
            start_scan = tip_ap + 10;
        otherwise
            print('Choose Undeformed or Deformed for argument')
    end

    t_pt = 0;
    prevt_pt = 0;

    % Searches through columns to obtain points of the tip
    for ii = start_scan:-1:1

        if ii <= (tip_ap - 250)
            print("Tip not found. Redo image capture")
            tip = size(Gray2,2);
            break;
        end

        % Searches through rows to obtain points of the tip
        for jj = 1:size(Gray2,1)
            if Gray2(jj,ii) == 1  
                t_pt = t_pt + 1;
            end        
        end

        if t_pt >= t_pt_th && prevt_pt >= t_pt_th
            tip = ii;
            break;
        else
            prevt_pt = t_pt;
            t_pt = 0;
        end
    end 

end


%% Function: Finding the start and end points

function [start_y, end_y, top_d, bot_d] = y_diff(Gray2, start_off, tip)
    prev_y1 = 1;
    prev_y2 = 1;
    tol_min = 3;          % Minimum space
    tol_max = 60;         % Space between points from which to consider

    start_yf = zeros(1,size(Gray2,1));
    end_yf = zeros(1,size(Gray2,1));

    % Goes through columns to obtain points of the edges
    for ii = 1:size(Gray2,1)
        if Gray2(ii,start_off) == 1 && tol_min <= abs(ii - prev_y1) && ...
                abs(ii - prev_y1) <= tol_max
            start_yf(ii) = ii;
            prev_y1 = ii;
        end
        if Gray2(ii,tip) == 1 && tol_min <= abs(ii - prev_y2) && ...
                abs(ii - prev_y2) <= tol_max
            end_yf(ii) = ii;
            prev_y2 = ii;
        end    
    end 

    % Gets the outermost y coordinates of the two strips
    start_y(1) = min(start_yf(start_yf~=0));
    start_y(2) = max(start_yf(start_yf~=0));
    end_y(1) = min(end_yf(end_yf~=0));
    end_y(2) = max(end_yf(end_yf~=0));
    
    % start_y
    % end_y
    
    % Calcalates the difference in y heights of the strips
    top_d = abs(end_y(1) - start_y(1));
    bot_d = abs(end_y(2) - start_y(2));
end 



%% Function: Aquiring snapshot

function Gray2 = out_find(cam, ltest, p2mm)

    img_test = snapshot(cam,'immediate');
    
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

