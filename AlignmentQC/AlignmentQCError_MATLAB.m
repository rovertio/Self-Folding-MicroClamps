%% Alignment QC Error Analysis

off_dis = 5;            % Length apart for edge identification
p2mm = 10;
th_an = 0.875/13;

Err_an = @(t) atan(((off_dis*tan(t)) + (2/p2mm))/off_dis) - t;
an_range = linspace(0,0.1);

figure(1)
hold on
% subplot(1,2,1)
% plot((180/pi)*an_range, (180/pi)*Err_an(an_range))
% xlabel('Angle of clamp jaws (deg)')
% ylabel('Angle measurement error (deg)')
% xline((180/pi)*th_an)
% hold off
% 
% subplot(1,2,2)
% hold on
plot((180/pi)*an_range, 13*Err_an(an_range), 'LineWidth', 2, 'Color',...
    'blue', 'DisplayName', 'Error due to resolution')
xlabel('Angle of clamp jaws (deg)')
ylabel('Alignment measurement error (mm)')
xline((180/pi)*th_an,  ':', 'DisplayName', "Error at threshold angle", ...
    "LineWidth", 1, "Color", [1,0,1])
title(['Error in distance reading vs angle between jaws ' ...
    '(resolution = 10pixels/mm)'])

legend show
hold off

%%

syms v_ref v_check
% v_check = [1,1];
theta = @(v_ref) acos(dot(v_ref, v_check)/(norm(v_ref)*norm(v_check)));
diff(acos(dot(v_ref, v_check)/(norm(v_ref)*norm(v_check))))



