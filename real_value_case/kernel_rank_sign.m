function [kernel_vect,u] = kernel_rank_sign(y,T)

[N, K] = size(y);

IN_T = inv(T);

SR_IN_T = sqrtm(IN_T);
Q = dot(SR_IN_T*y,SR_IN_T*y);
u = SR_IN_T*y./sqrt(Q);


[~,p] = sort(Q,'ascend');
r = 1:K;
r(p) = r;

kernel_vect = -chi2inv(r/(K+1),N)/2;

end

