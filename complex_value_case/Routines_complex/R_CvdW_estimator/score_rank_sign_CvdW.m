function [score_vect,u,SR_IN_T] = score_rank_sign_CvdW(y,T)

[N, K] = size(y);

% Evaluation of the square root of the inverse of the preliminary estimator T
SR_IN_T = inv(sqrtm(T));

% Evaluation of Q^\star in Eq. (48)
Q = dot(SR_IN_T*y,SR_IN_T*y);

% Evaluation of u^\star in Eq. (49)
u = SR_IN_T*y./sqrt(Q);

% Evaluation of the ranks r^\star of Q^\star
[~,p] = sort(Q,'ascend');
r = 1:K;
r(p) = r;

% Evaluation of the "complex" van der Waerden score function in Eq. (50)
score_vect = gaminv(r/(K+1),N,1);

end

