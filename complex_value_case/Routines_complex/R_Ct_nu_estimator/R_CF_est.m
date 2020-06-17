function [ N_VDW,  beta_est] = R_CF_est( y, T, nu, pert)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: 
% Complex-valued data matrix: y (The data is assumed to be zero-mean observations. 
                  % If it doesn't, y has to be centered using 
                  % a preliminary estimator of the location parameter)
% Preliminary consistent estimator: T 
% Perturabtion parameter: pert
% Parameter of the t-score function: nu

% Output:
% Semiparametric efficient R-estimator of the shape: N_VCW
% Convergence parameter: beta_est

% The score function used here is the one obtained from the t-distribution.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The preliminary estimator has to satisfy the constraint [T]_{1,1}=1
T = T/T(1,1);

[N, K] = size(y);

% Definition of the "small perturbation" matrix 
V = pert*(randn(N,N) +1i*randn(N,N));
V = (V+V')/2;
V(1,1)=0;

% Estimation of alpha
[alpha_est, Delta_T, Psi_T] = alpha_estimator_sub_CF( y, T, V, nu);

beta_est = 1/alpha_est;

% Vectorized one-step estimatimator of the shape matrix
N_VDW_vec = T(:) + [0; beta_est*(Psi_T\Delta_T)/sqrt(K)];

% One-step estimatimator of the shape matrix
N_VDW = reshape(N_VDW_vec, [N,N]);


end
