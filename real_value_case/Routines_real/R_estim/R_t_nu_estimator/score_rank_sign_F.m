function [score_vect,u,SR_IN_T] = score_rank_sign_F(y,T,nu)

[N, K] = size(y);

% Evaluation of the square root of the inverse of the preliminary estimator T
SR_IN_T = inv(sqrtm(T));

% Evaluation of Q^\star in Eq. (26)
Q = dot(SR_IN_T*y,SR_IN_T*y);

% Evaluation of u^\star in Eq. (27)
u = SR_IN_T*y./sqrt(Q);

% Evaluation of the ranks r^\star of Q^\star
[~,p] = sort(Q,'ascend');
r = 1:K;
r(p) = r;

% Evaluation of the t_nu-score function in Eq. (35)
f_inv_appo = finv(r/(K+1),N,nu);
psi_0_fun = N*(N + nu)./(nu + N*f_inv_appo);
score_vect = f_inv_appo.*psi_0_fun;
end

