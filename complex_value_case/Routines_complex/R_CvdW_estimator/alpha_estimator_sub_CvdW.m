function [alpha_est, Delta_T, Psi_T] = alpha_estimator_sub_CvdW(y, T, V)

[N, K] = size(y);

% Evaluation of the approximation of the efficient central sequence Delta_T in Eq. (47)
% and of the matrix Psi_T, where T is the preliminary estimator
[Delta_T, Psi_T] = Delta_Psi_eval_CvdW(y, T);

% Evaluation of the perturbed approximation of the efficient central sequence Delta_T
T_pert = T + V/sqrt(K);
Delta_T_pert = Delta_only_eval_CvdW(y, T_pert);

% Estimation of alpha (see Eq. (53))
V_1 = V(:);
alpha_est = norm(Delta_T_pert-Delta_T)/norm(Psi_T*(V_1(2:end)));

end

