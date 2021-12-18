clear all;
close all;
clc;
 
data = xlsread('dataset SeriesJ BoxJenkins');
GasRate = data(:,1);
CO2 = data(:,2);

for(i=1:size(data,1)-4)
    X1(i,1) = GasRate(i);
    X2(i,1) = GasRate(i+1);
    X3(i,1) = GasRate(i+2);
    X4(i,1) = GasRate(i+3);
    X5(i,1) = CO2(i+3);
    Y1(i,1) = CO2(i+4);
end

%Inputs and output
X = [X1 X2 X3 X4 X5];
Y = [Y1];
% Normalization
MM = minmax(X');
MM = [MM; minmax(Y')];
for(i=1:size(X1,1))
    X1(i) = (X1(i) - MM(1,1))/(MM(1,2) - MM(1,1));
    X2(i) = (X2(i) - MM(2,1))/(MM(2,2) - MM(2,1));
    X3(i) = (X3(i) - MM(3,1))/(MM(3,2) - MM(3,1));
    X4(i) = (X4(i) - MM(4,1))/(MM(4,2) - MM(4,1));
    X5(i) = (X5(i) - MM(5,1))/(MM(5,2) - MM(5,1));
    Y1(i) = (Y1(i) - MM(6,1))/(MM(6,2) - MM(6,1));
end
X = [X1 X2 X3 X4 X5];
Y = [Y1];
%inputs = Data(:,1:7);
output = Y;
x = X;
data = [x output];

[num_points, num_vars] = size(x);
basic_alpha = 0.01;
lambda = 0.05;
window_size = 45;
K_init = 0.1*eye(num_vars);
N_max = 5;
rho = 0.10;
v_r = 8; % valor de intervalo de tempo que irá analisar o erro - Somente no eFMI
num_training_points = num_points; 

tic;
[ erro, ys, r, num_clusters] = eFMI(x, output, basic_alpha, window_size , K_init,  num_training_points, rho, N_max, v_r);
toc;
x=toc;
tempo = x/num_points*1000;
mse = (1/num_points)*sum(r);
rmse = sqrt(mse)
ndei= rmse/std(output)
regras=num_clusters(num_points);
X55 = regras;
X56 = rmse;
X57 = tempo;
X58 = ndei;


%% Plota os pontos e os clusters (contorno)


figure;
y = output;
plot(y);
hold on;
plot(ys,'r-');
%title(sprintf('RMSE = %.4f - IEND = %.4f',rmse, ndei));
xlabel('Amostras');
ylabel('Saída');
legend('Saída Desejada','Saída do eFCE');


figure;
y = output;
plot(y);
hold on;
plot(ys,'r-');
%title(sprintf('RMSE = %.4f - IEND = %.4f',rmse, ndei));
xlabel('Samples');
ylabel('output');
legend('Desired output','eFCE output');

numeroregras = [0 max(num_clusters)+1];
evolu = [1 num_training_points];
regra1 = num_clusters;
figure;
plot(num_clusters);
xlim([1 num_training_points]);
ylim([0 max(num_clusters)+1]);
xlabel('Amostras');
ylabel('Número de Regras');


numeroregras = [0 max(num_clusters)+1];
evolu = [1 num_training_points];
regra1 = num_clusters;
figure;
plot(num_clusters);
xlim([1 num_training_points]);
ylim([0 max(num_clusters)+1]);
xlabel('Samples');
ylabel('Number of Rules');

