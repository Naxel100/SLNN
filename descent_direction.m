function dk = descent_direction(wk, g_act, g_ant, H, isd, icg, irc, nu, d_ant, k)
    betak = 0;

    % Métode del gradiente
    if isd == 1 || (isd == 2 && k == 1), dk = -g_act;

    % Método del gradiente conjugado
    elseif isd == 2
        % Si es la primera iteración o tenemos que reiniciar betak = 0
        if ~(irc == 1 && mod(k, length(wk(:, 1))) == 0) && ~(irc == 2 && abs(g_act' * g_ant / norm(g_act)^2) >= nu)
            % Fletcher-Reeves
            if icg == 1, betak = norm(g_act)^2 / norm(g_ant)^2;
            % Polak-Ribière
            else, betak = max(g_act' * (g_act - g_ant) / norm(g_ant)^2, 0);
            end
        end
        dk = -g_act + betak * d_ant;

    % BFGS
    elseif isd == 3, dk = -H * g_act;

    % Método del gradiente estocástico
    elseif isd == 7, dk = -g_act;
    end
end
