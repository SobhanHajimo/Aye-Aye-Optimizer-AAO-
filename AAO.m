% ==================================================================================================================================================
% Algorithm Name: Aye-Aye Optimizer (AAO)
% Version: 1.0
% Author: Sobhan Hajmohammadi
% Contact Email: sobhanhajimo@gmail.com
%
% Description:
% MATLAB implementation of the Aye-Aye Optimizer (AAO), a bio-inspired metaheuristic algorithm based on the percussive foraging strategy.
%
% Reference (Please cite if you use this code):
% Hajmohammadi, S., Mahmoodabadi, M.J., Saremi, S. et al. Aye-Aye Optimizer (AAO): A Bio-Inspired Metaheuristic Algorithm Based on the Percussive Foraging Strategy. Int J Comput Intell Syst (2026).
% https://doi.org/10.1007/s44196-026-01608-1
% ==================================================================================================================================================

function P_opt_fitness = AAO

y = @(x) F1(x);

max_iter = 10000;
N_t = 10;
N_d = 20;
lb = -100;
ub = 100;
D_max = (ub - lb)/2;

P_i = rand(N_t, N_d) .* (ub - lb) + lb;
fitness = zeros(1, N_t);

for i = 1:N_t
    fitness(1,i) = y(P_i(i, :));
end

[best_fitness_iter, idx] = min(fitness(1,:));
P_ref(1, :) = P_i(idx, :);
P_ref_fitness(1, :) = best_fitness_iter;

[best_fitness_global, idx] = min(P_ref_fitness);
P_opt = P_ref(idx, :);
P_opt_fitness = best_fitness_global;

alpha_c = 1.55;
beta_c = 1.45;
gamma_c = 0.5;

for iter = 1:max_iter

    R = exp(-iter);

    for i = 1:N_t

        distance_to_target = norm(P_i(i, :) - P_opt);
        rand_factor = rand(1, N_d) - 0.5;
        alpha = alpha_c * (1 - exp(-iter / max_iter) + rand_factor);
        beta = (beta_c + (0.1 / (1 + exp((max_iter / 2 - iter))))) * rand_factor;
        gamma = gamma_c * (iter / max_iter) * (1 - distance_to_target / D_max);

        P_i_new(i, :) = P_i(i, :) + (alpha + beta + gamma) .* (P_opt - P_i(i, :));

        if or(max(P_i_new(i, :)) > ub, min(P_i_new(i, :)) < lb)
            P_i_new(i, :) = P_i(i, :) + R * (P_ref(iter, :) - P_i(i, :));
        end

        fitness_new = y(P_i_new(i, :));
        if fitness_new < fitness(1,i)
            fitness(1,i) = fitness_new;
            P_i(i, :) = P_i_new(i, :);
        end
    end

    [best_fitness_iter, idx] = min(fitness(1,:));
    P_ref(iter + 1, :) = P_i(idx, :);
    P_ref_fitness(iter + 1, :) = best_fitness_iter;

    [best_fitness_global, idx] = min(P_ref_fitness);
    P_opt = P_ref(idx, :);
    P_opt_fitness = best_fitness_global;
end

figure;
semilogy(P_ref_fitness);
title('AAO Convergence Curve');
xlabel('Iteration');
ylabel('Fitness');
grid on;
box on;

end