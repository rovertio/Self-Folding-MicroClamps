%% Spring analysis code
% Code used to compute the values needed for the force quality control
% method using springs. Derivation of euqations used can be found in our
% site

clear all;

%% Calculation of dependent physical/geometrical values
% Also converts units for consistency

% Input values for pixel conversion and length tested
pix_count = 231;                    % pixels counted per real inch
% Conversion of values
p2mm = ceil(pix_count/(25.4));      % pixels per mm

% Threshold force
th_f = 0.45;                               % Threshold force (N)
N2gf = 101.972;                           % Newtons to gram-force
th_f = th_f*N2gf;                         % Threshold force (gf)

% Material properties (Stainless steel)
E = 200;           % Modulus of elasticity (GPa)
% Material properties (Stainless steel)
E = E*10^9;        % Modulus of elasticity (GPa)

% Lengths
% L1 = L1*0.0254;    % Length of first arm (m)
% L2 = L2*0.0254;    % Length of second arm (m)


% Assuming rectangular cross section
h = 0.635;          % Thickness of metal (mm)
b = 3.8;            % Width of strip (mm)
delta = 0.865;      % Deflection of spring jaws (mm)

% Assuming rectangular cross section
delta = 0.865*0.001;% Deflection of spring jaws (mm)
h = h*0.001;        % Thickness of metal (m)
b = b*0.001;        % Width of strip (m)
CA = h*b;           % Cross sectional area (m^2)
I1 = (b*h^3)/12;    % Inertia of rectangular area (m^4)



%% Error of the Force in relation with the length of spring used

% Resolution calculation (meters per pixel)
pix_res = (p2mm^(-1))*0.001;
% Error in strip mesurement (mm)
mes_res = 1;

% Range of test values
L_testvalues = linspace(30, 80, 51);

% Resolution force calculations assuming cantilever connection
P_force = @(L) N2gf.*((pix_res.*3.*E.*I1)./((L*0.001).^3));
P_test = P_force(L_testvalues);
% Psotive tolerance of measurement error
P_upmes = P_force(L_testvalues + 1);
%Negative toelrance of measurement error
P_lowmes = P_force(L_testvalues - 1);

% Resolution length calculation from force (gf) to legngth (mm)
L_bound = @(P) 1000.*((3.*pix_res.*E.*(I1).*((P./N2gf).^(-1))).^(1/3));

% Strip length calculation from force (gf) to legngth (mm)
L_strip = @(P) 1000.*((3.*delta.*E.*(I1).*((P./N2gf).^(-1))).^(1/3));


% Plot of the potential error in readings
figure(1);
hold on 
% Curve from nominal distance values
pnom = plot(L_testvalues, P_test, 'LineWidth', 2, 'Color', 'blue',...
    'DisplayName', 'Nominal calculated force');
% Curve from upper measurement error
pup = plot(L_testvalues, P_lowmes, 'LineStyle', ':',...
    'LineWidth', 1, 'Color', [1,0,0],...
    'DisplayName', 'Upper bound calculated force');
% Curve from lower measurement error
plow = plot(L_testvalues, P_upmes, 'LineStyle', ':',...
    'LineWidth', 1, 'Color', [0,1,1],...
    'DisplayName', 'Lower bound calculated force');

% Error at different benchmark percentages
xline(L_bound(0.1*th_f), ':', "10% (gf): " + th_f*0.10);
xline(L_bound(0.05*th_f), ':', "5% (gf): " + th_f*0.05);
% xline(L_bound(0.01*th_f), ':', "1% (gf): " + th_f*0.01);

% Error at calculated lengths for upper and lower bounds
xline(L_strip(th_f), ':', "Upper Bound (45gf)", ...
    "LineWidth", 1, "Color", [1,0,1]);
xline(L_strip(25), ':', "Lower Bound (25gf)", ...
    "LineWidth", 1, "Color", [1,0,1]);

xlabel('Test Strip Lengths (mm)')
ylabel('Error in force reading (gf)')
title(['Error in force reading vs strip test length ' ...
    '(resolution = 10pixels/mm)'])

legend([pnom, pup, plow], 'Nominal calculated force',...
    'Upper bound calculated force', 'Lower bound calculated force')
hold off



%% Iteration to find ideal lengths of strips

con_crit = 0.01;
itt_err = 10;
up_err = 10;
low_err = 10;

up_bound = 45;
low_bound = 25;
up_len = 45;
low_len = 55;

% Finding length of strip for lower bound testing
while itt_err > conv_crit
    
end





