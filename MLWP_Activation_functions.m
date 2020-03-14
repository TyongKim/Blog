% This script is plot various activation functions used in deep learning
% Developed by Taeyong Kim from the Seolu National University
% chs5566@snu.ac.kr
% March 14, 2020

clear; clc; close all

% Sigmoid function
x = -10:0.01:10;
y = 1./(1+exp(-x));

figure()
set(gcf, 'Position',  [100, 100, 500, 200])
plot(x,y,'k','LineWidth',2)
grid on

% Various activation function
ReLU = zeros([length(x),1]);
LeakyReLU = zeros([length(x),1]);
ELU = zeros([length(x),1]);
for ii =1:length(x)
    tmp = x(ii);
    
    ReLU(ii,1) = max([0,tmp]);
    LeakyReLU(ii,1) = max([0,tmp])+0.01*min([0,tmp]);
    ELU(ii,1) = max([0,tmp])+min([0,1*(exp(x)-1)]);
    
end

Tanh = (exp(x)-exp(-x))./(exp(x)+exp(-x));
Softplus = 1./1.*log(1+exp(1*x));

% Plotting
figure()
set(gcf, 'Position',  [100, 100, 500, 200])
plot(x, ReLU, 'k', 'LineWidth',2)
grid on

figure()
set(gcf, 'Position',  [100, 150, 500, 200])
plot(x, LeakyReLU, 'k', 'LineWidth',2)
grid on

figure()
set(gcf, 'Position',  [100, 200, 500, 200])
plot(x, ELU, 'k', 'LineWidth',2)
grid on

figure()
set(gcf, 'Position',  [100, 250, 500, 200])
plot(x, Tanh, 'k', 'LineWidth',2)
grid on

figure()
set(gcf, 'Position',  [100, 300, 500, 200])
plot(x, Softplus, 'k', 'LineWidth',2)
grid on
