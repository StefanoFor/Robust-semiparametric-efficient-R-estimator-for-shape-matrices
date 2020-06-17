function [ N_F,  beta_est] = R_F_est( y, T, nu, pert)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: 
% Real-valued data matrix: y (The data is assumed to be zero-mean observations. 
                  % If it doesn't, y has to be centered using 
                  % a preliminary estimator of the location parameter)
% Preliminary consistent estimator: T 
% Perturabtion parameter: pert

% Output:
% Semiparametric efficient R-estimator of the shape: N_F
% Convergence parameter: beta_est

% The score function used here is the one obtained from the t_nu-distribution.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The preliminary estimator has to satisfy the constraint [T]_{1,1}=1
T = T/T(1,1);

[N, K] = size(y);

% Definition of the Duplication and Elimination matrices
D_n = full(DuplicationM(size(T)));
M_n = D_n(:,2:end).';

E_n = full(EliminationM(size(T)));
N_n = E_n(2:end,:);

% Definition of the "small perturbation" matrix 
V = pert*randn(N,N);
V = (V+V')/2;
V(1,1)=0;

% Estimation of alpha
[alpha_est, Delta_T, Psi_T] = alpha_estimator_sub_F( y, T, V, nu, M_n, N_n);

beta_est = 1/alpha_est;

% Vectorized one-step estimatimator of the shape matrix
N_F_vec = E_n*T(:) + [0; beta_est*(Psi_T\Delta_T)/sqrt(K)];

% One-step estimatimator of the shape matrix
N_F = reshape(D_n*N_F_vec, [N,N]);

end