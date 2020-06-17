function Delta_T = Delta_only_eval_CF(y, T, nu)

[N, K] = size(y);

% Evaluation of the score function and of the vector u
[score_vect,u,inv_sr_T] = score_rank_sign_CF(y,T, nu);

% Evaluation of the matrix L_V in Eq. (43)
inv_sr_T2 = kron(inv_sr_T.',inv_sr_T);
I_N = eye(N);
J_n_per = eye(N^2) - I_N(:)*I_N(:).'/N;
K_V = (inv_sr_T2*J_n_per);

%%%% Evaluation of the approximation of the efficient central sequence in Eq. (47)
%%% Pedagogical version of the calculation
% Score_appo = zeros(N^2,1);
% for k=1:K
%    Mat_appo = (u(:,k)*u(:,k)');
%    Score_appo = Score_appo  + score_vect(k)*Mat_appo(:);
% end

%%% Fast version of the calculation
Mat_appo = u .* reshape( u', [1 K N] );
Mat_appo = reshape( permute( Mat_appo, [1 3 2] ), N^2, [] );
Score_appo = Mat_appo*score_vect.';

% Approximated efficient central sequence in Eq. (47)
Delta_T = K_V*Score_appo/sqrt(K);
Delta_T = Delta_T(2:end);

end
