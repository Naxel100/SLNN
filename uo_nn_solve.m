% wo = w óptima, fo = valor de la función en el mínimo, tr_acc = precisión de training
% te_acc = precisión de test, niter = número de iteraciones, tex = tiempo de ejecución después de generar el dataset.

function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_ga1,sg_al0,sg_ga2,icg,irc,nu)
    
    % Generación de datsets
    [Xtr, ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
    [Xte, yte] = uo_nn_dataset(te_seed, te_q, num_target, -1);

    % Definición de las funciones necesarias
    sig = @(X) 1 ./ (1 + exp(-X));                                                       % Sigmoide
    y = @(X, w) sig(w' * sig(X));                                                        % Respuesta neurona salida
    L = @(w) norm(y(Xtr, w) - ytr)^2 + (la * norm(w)^2)/2;                               % Loss function
    gL = @(X, Y, w) 2 * sig(X) * ((y(X, w) - Y) .* y(X, w) .* (1 - y(X, w)))' + la * w;  % Gradiente
    g = @(w) gL(Xtr, ytr, w);                                                            % Gradiente con datos de training
    acc = @(Xds,yds,wo) 100*sum(yds==round(y(Xds,wo)))/length(yds);                      % Accuracy
    %
    
    % Inicialización variables
    w0 = zeros(35, 1); n = length(w0); k = 0; m = floor(sg_ga1*tr_p);
    
    wk = zeros(n, kmax); d_act = zeros(n, 1); H_act = eye(n);    % Nos guardamos todas las wk por dónde pasamos y los iWk
    iWk = zeros(1, kmax); al_act = 0; wk(:, 1) = w0; g_ant = 1;  % necesarios para el estudio posterior de los resultados.
                                                                 % Para todo lo demás solo necesitamos el antrior y el actual.
                                                                 
    t1 = clock;  % Inicio algoritmo
    
    
    % ********************* Algoritmo optimización ********************* %
    
    if isd ~= 7, g_act = g(w0);
    else
        batch = randi(tr_p, 1, m);
        Xtrs = Xtr(:, batch); ytrs = ytr(batch);
        g_act = gL(Xtrs, ytrs, w0);
    end
    
    while k < kmax && norm(g_ant) > epsG
        k = k + 1;
        d_ant = d_act; H_ant = H_act; al_ant = al_act; w_act = wk(:, k);
        
        % Cálculo del descent direction dado el método
        d_act = descent_direction(wk, g_act, g_ant, H_act, isd, icg, irc, nu, d_ant, k);

        % Cálculo de la alfa dada la descent direction
        [al_act, iWk(k)] = find_alpha(L,g,g_act,g_ant,w_act,d_act,c1,c2,kmaxBLS,epsal,al_ant,k,wk,d_ant,ialmax,sg_ga2,sg_al0,kmax,isd);
        
        % Actualización de los vecotres
        wk(:, k + 1) = w_act + al_act*d_act;
        
        g_ant = g_act;   % Actualización gradiente
        if isd ~= 7, g_act = g(wk(:, k + 1));
        else
            batch = randi(tr_p, 1, m);
            Xtrs = Xtr(:, batch); ytrs = ytr(batch);
            g_act = gL(Xtrs, ytrs, wk(:, k + 1));
        end
        
        if isd == 3    % Actualización matriz H en BFGS
            y = g_act - g_ant; s = wk(:, k + 1) - w_act; rhok = 1 / (y' * s);
            H_act = (eye(n) - rhok*s*y') * H_ant * (eye(n) - rhok*y*s') + rhok*(s*s');
        end
    end
    
    % ******************************************************************* %
    
    t2 = clock;
    
    iWk = iWk(1:k); iWk(k) = NaN; wk = wk(:, 1:k); niter = k;
    wo = wk(:, k);  fo = L(wo); tex = etime(t2,t1);
    tr_acc = acc(Xtr,ytr,wo); te_acc = acc(Xte,yte,wo);
end