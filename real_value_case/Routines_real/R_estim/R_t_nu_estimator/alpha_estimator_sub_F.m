function [alpha_est, Delta_T, Psi_T] = alpha_estimator_sub_F(y, T, V, nu, M_n, N_n)

[N, K] = size(y);

% Evaluation of the approximation of the efficient central sequence Delta_T in Eq. (33)
% and of the matrix Psi_T, where T is the preliminary estimator
[Delta_T, Psi_T] = Delta_Psi_eval_F(y, T, nu, M_n);

% Evaluation of the perturbed approximation of the efficient central sequence Delta_T
T_pert = T + V/sqrt(K);
Delta_T_pert = Delta_only_eval_F(y, T_pert, nu, M_n);

% Estimation of alpha (see Eq. (37))
V_1 = V(:);
alpha_est = norm(Delta_T_pert-Delta_T)/norm(Psi_T*(N_n*V_1));
end

