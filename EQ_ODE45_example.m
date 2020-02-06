%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is developed for explaining how to use ODE45 in MATLAB
% Developed by Taeyong Kim from the Seoul National Unversity.
% chs5566@snu.ac.kr
% Feb 6 ,2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc
close all


% First example
[t,y] = ode45(@(t, y) 2*y,[0, 3],1);  % Numericla solution
y_real = exp(2*t);                    % Analytic solution

% plot
plot(t, y, 'b')
hold on
plot(t,y_real, 'r--')
xlabel('t')
ylabel('y')


% Second example
m=1; % mass
c=1; % damping
k=1; % stiffness

ftt = 0:0.01:10;
ft = sin(ftt); % external force

% 'EQ_func_ode_2.m' file needs to placed in the working directory 
[t2,y2] = ode45(@(t2,y2) EQ_func_ode_2(t2,y2, ftt, ft, m, c, k), [0, 10],[0, 0]);

%plot
figure()
plot(t2,y2(:,1))
xlabel('t')
ylabel('u')
