function [score_vect,u,SR_IN_T] = score_rank_sign_vdW(y,T)

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

% Evaluation of the van der Waerden score function in Eq. (34)
score_vect = chi2inv(r/(K+1),N)/2;

end

