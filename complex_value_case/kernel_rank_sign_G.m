function [kernel_vect,u] = kernel_rank_sign_G(y,T)

[N, K] = size(y);

IN_T = pinv(T);

SR_IN_T = sqrtm(IN_T);
Q = dot(SR_IN_T*y,SR_IN_T*y);
u = SR_IN_T*y./sqrt(Q);


[~,p] = sort(Q,'ascend');
r = 1:K;
r(p) = r;

kernel_vect = -gaminv(r/(K+1),N,1);

end

