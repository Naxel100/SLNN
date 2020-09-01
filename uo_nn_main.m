clear;
clc;
%
% Parámetros para la generación de datos
%
num_target = [1];       % Número target.
tr_freq    = .5;        % Frecuencia del número target en el training set.
tr_p       = 250;       % Número de imágenes de entrenamiento.
te_q       = 250;       % Número de imágenes de test.
tr_seed    = 123456;    % Training seed.
te_seed    = 789101;    % Test seed.

%
% Parámetros para el programa de optimización
%
la = 0.0;                                                     % Regularización L2.
epsG = 10^-6; kmax = 1000;                                    % Criterio de parada.
ils=3; ialmax = 2; kmaxBLS=30; epsal=10^-3;c1=0.01; c2=0.45;  % Linesearch.
isd = 7; icg = 2; irc = 2 ; nu = 1.0;                         % Search direction.
sg_ga1 = 0.05; sg_al0 = 2; sg_ga2 = 0.3;                      % Gradiente estocástico

%
% Optimización
%
t1=clock;
[Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_ga1,sg_al0,sg_ga2,icg,irc,nu);
t2=clock;
fprintf('Wall time = %6.1d s.\n', etime(t2,t1));
uo_nn_Xyplot(Xtr,ytr,wo);
fprintf("niter: %d\n",niter);


