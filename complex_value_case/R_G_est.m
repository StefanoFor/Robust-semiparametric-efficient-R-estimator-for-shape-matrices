function [ N_VDW,  beta_est] = R_G_est( y, T, pert)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Imput: 
% Complex data matrix: y (Here the data is asumed to be zero-mean observations. 
                  % If it doesn't, y has to be centered using 
                  % a preliminary estimator of the location parameter)
% Preliminary estimator: T (with the constraint that [V]_{1,1}=1)
% Perturabtion parameter: pert

% Output:
% Semiparametric efficient R-estimator of the shape: N_VCW
% Convergence parameter: beta_est

% The score function used here is the van der Waerden score.

[N, K] = size(y);

V = pert*(randn(N,N) +1i*randn(N,N));
V = (V+V')/2;

[alpha_est, Delta_T, Psi_T] = alpha_estimator_sub_G( y, T, V);

beta_est = 1/alpha_est;

N_VDW_vec = T(:) + [0; beta_est*(Psi_T\Delta_T)/sqrt(K)];

N_VDW = reshape(N_VDW_vec, [N,N]);


end
