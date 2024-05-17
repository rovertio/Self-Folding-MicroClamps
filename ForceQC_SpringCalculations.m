%% Spring analysis code
% Code used to compute the values needed for the force quality control
% method using springs. Derivation of euqations used can be found in our
% site

clear all;

%% Definition of physical constants (Input)

% Threshold force
th_f = 0.3;         % Threshold force (N)

% Material properties (Stainless steel)
E = 200;            % Modulus of elasticity (GPa)
        
% Angles
a1 = 35;            % Angle between first juncture (deg)
b1 = 90;            % Angle of second bend (deg)

% Lengths
L3 = 0.1;           % Length of first arm (in)
L2 = 0.1;           % Length of second arm (in)

% Assuming rectangular cross section
h = 0.007;          % Thickness of metal (inches)
b = 0.079;          % Width of strip (inches)


%% Calculation of dependent physical/geometrical values
% Also converts units for consistency

% Material properties (Stainless steel)
E = 200*10^9;       % Modulus of elasticity (GPa)
        
% Angles
a1 = 35*(180/pi);   % Angle between first juncture (rad)
b1 = 90*(180/pi);   % Angle of second bend (rad)
g1 = (a1/2) + b1;   % Angle of third bend (rad)

% Lengths
L3 = 0.1*0.0254;    % Length of first arm (m)
L2 = 0.1*0.0254;    % Length of second arm (m)

% Assuming rectangular cross section
h = 0.007*0.0254;   % Thickness of metal (m)
b = 0.079*0.0254;   % Width of strip (m)
CA = h*b;           % Cross sectional area (m^2)
I1 = (b*h^3)/12;    % Inertia of rectangular area (m^4)


%% Displacement calculation formulas


