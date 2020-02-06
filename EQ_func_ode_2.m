%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is developed for explaining how to use ODE45 in MATLAB
% Developed by Taeyong Kim from the Seoul National Unversity.
% chs5566@snu.ac.kr
% Feb 6 ,2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Res = EQ_func_ode_2(t,y, ftt, ft, m, c, k)
   
    ft2 = interp1(ftt, ft, t);

    Res = [y(2); 
        
           -c/m*y(2) - k/m*y(1) + ft2;];
end