function [Delta_T, Psi_T] = Delta_Psi_eval_G(y, T)

[N, K] = size(y);

[kernel_vect,u] = kernel_rank_sign_G(y,T);

T2 = kron(T.',T);

sr_T2 = sqrtm(T2);

I_N = eye(N);

J_n_per = eye(N^2) - I_N(:)*I_N(:).'/N;

K_V = (sr_T2\J_n_per);

% Kernel_appo = zeros(N^2,1);
% for k=1:K
%    Mat_appo = (u(:,k)*u(:,k)');
%    Kernel_appo = Kernel_appo  + kernel_vect(k)*Mat_appo(:);
% end

Mat_appo = u .* reshape( u', [1 K N] );
Mat_appo = reshape( permute( Mat_appo, [1 3 2] ), N^2, [] );
Kernel_appo = Mat_appo*kernel_vect.';

Delta_T = -(sr_T2\J_n_per)*Kernel_appo/sqrt(K);
Delta_T = Delta_T(2:end);

K_c = K_V*K_V';
Psi_T = K_c(2:end,2:end);

end

