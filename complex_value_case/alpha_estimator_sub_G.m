function [alpha_est, Delta_T, Psi_T] = alpha_estimator_sub_G(y, T, V)

V(1,1)=0;

[N, K] = size(y);

[Delta_T, Psi_T] = Delta_Psi_eval_G(y, T);

T_pert = T + V/sqrt(K);

Delta_T_pert = Delta_only_eval_G(y, T_pert);

V_1 = V(:);

alpha_est = norm(Delta_T_pert-Delta_T)/norm(Psi_T*(V_1(2:end)));

end

