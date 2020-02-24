function [alpha_est, Delta_T, Psi_T] = alpha_estimator_sub(y, T, V, M_n, N_n)

V(1,1)=0;

[N, K] = size(y);

[Delta_T, Psi_T] = Delta_Psi_eval(y, T, M_n);

T_pert = T + V/sqrt(K);

Delta_T_pert = Delta_only_eval(y, T_pert, M_n);

V_1 = V(:);

alpha_est = norm(Delta_T_pert-Delta_T)/norm(Psi_T*(N_n*V_1));

end

