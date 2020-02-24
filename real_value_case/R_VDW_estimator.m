function [ N_VDW,  beta_est] = R_VDW_estimator( y, T, pert)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Imput: 
% Real-valued data matrix: y (Here the data is asumed to be zero-mean observations. 
                  % If it doesn't, y has to be centered using 
                  % a preliminary estimator of the location parameter)
% Preliminary estimator: T (with the constraint that [V]_{1,1}=1)
% Perturabtion parameter: pert

% Output:
% Semiparametric efficient R-estimator of the shape: N_VCW
% Convergence parameter: beta_est

% The score function used here is the van der Waerden score.

[N, K] = size(y);

D_n = full(DuplicationM(size(T)));
M_n = D_n(:,2:end).';

E_n = full(EliminationM(size(T)));
N_n = E_n(2:end,:);

V = pert*randn(N,N);
V = (V+V')/2;

[alpha_est, Delta_T, Psi_T] = alpha_estimator_sub( y, T, V, M_n, N_n);

beta_est = 1/alpha_est;

N_VDW_vec = E_n*T(:) + [0; beta_est*(Psi_T\Delta_T)/sqrt(K)];

N_VDW = reshape(D_n*N_VDW_vec, [N,N]);


end
