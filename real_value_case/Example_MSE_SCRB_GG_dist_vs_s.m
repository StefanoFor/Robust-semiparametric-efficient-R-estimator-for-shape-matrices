clear all
close all
clc


Ns=10^6; % monte carlo runs
N = 8;
Max_it = 50;

ro=0.8;
sigma2 = 4;

perturbation_par = 0.01;
nu_par = 5;

svect=[0.1:0.1:2];

K = 5*N;
Nl=length(svect);
n=[0:N-1];

rx=ro.^n; % Autocorrelation function
Sigma = toeplitz(rx);
mu = ones(N,1);

L=chol(Sigma);
L=L';

D_n = full(DuplicationM(size(Sigma)));
pinv_D_n = pinv(D_n);

L_n = full(EliminationM(size(Sigma)));
Shape_S = N*Sigma/trace(Sigma);
Inv_Shape_S = inv(Shape_S);
theta_true = L_n*(Shape_S(:));

J_phi = jacobian_constraint(N);
U = null(J_phi');

DIM = N*(N+1)/2;
tic
for il=1:Nl
    
    s=svect(il)
    b = ( sigma2*N*gamma(N/(2*s))/( 2^(1/s)*gamma( (N+2)/(2*s) ) ) )^s;
    
    MSE_SCM = zeros(DIM,DIM);
    MSE_NTy = zeros(DIM,DIM);
    MSE_RM = zeros(DIM,DIM);
    MSE_RF = zeros(DIM,DIM);
    bias_SCM = zeros(DIM,1);
    bias_NTy = zeros(DIM,1);
    bias_RM = zeros(DIM,1);
    bias_RF = zeros(DIM,1);
    
    parfor ins=1:Ns
        
        w = randn(N,K);
        w_norm = sqrt(dot(w,w));
        w_n = w./repmat(w_norm,N,1);
        x = L*w_n;
        R = gamrnd(N/(2*s),2*b,1,K);
        y = repmat(R,N,1).^(1/(2*s)).*x;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SCM
        SCM = y*y'/K;
        Scatter_SCM = N*SCM/trace(SCM);
        err_v = L_n*Scatter_SCM(:)-theta_true;
        bias_SCM = bias_SCM + err_v/Ns;
        err_MAT = err_v(1:end,:)*err_v(1:end,:)';
        MSE_SCM = MSE_SCM + err_MAT/Ns;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tyler matrix estimator
        Ty = Tyler_R_11( y, Max_it);
        NTy = N*Ty/trace(Ty);
        err_v = L_n*NTy(:)-theta_true;
        bias_NTy = bias_NTy + err_v/Ns;
        err_NTy = err_v(1:end,:)*err_v(1:end,:)';
        MSE_NTy= MSE_NTy + err_NTy/Ns;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% R - estimators
        
        % R-estimator with the van der Waerden score function
        [RM, a] = R_vdW_est( y, Ty, perturbation_par);
        RM = N*RM/trace(RM);
        err_v = L_n*RM(:)-theta_true;
        bias_RM = bias_RM + err_v/Ns;
        err_RM = err_v(1:end,:)*err_v(1:end,:)';
        MSE_RM= MSE_RM + err_RM/Ns;
        
        % R-estimator with the t_\nu score function
        [RF, a] = R_F_est( y, Ty, nu_par, perturbation_par);
        RF = N*RF/trace(RF);
        err_v = L_n*RF(:)-theta_true;
        bias_RF = bias_RF + err_v/Ns;
        err_RF = err_v(1:end,:)*err_v(1:end,:)';
        MSE_RF= MSE_RF + err_RF/Ns;

    end
    Fro_MSE_SCM(il) = norm(MSE_SCM,'fro');
    Fro_MSE_NTy(il) = norm(MSE_NTy,'fro');
    Fro_MSE_RM(il) = norm(MSE_RM,'fro');
    Fro_MSE_RF(il) = norm(MSE_RF,'fro');
    L2_bias_SCM(il) = norm(bias_SCM);
    L2_bias_NTy(il) = norm(bias_NTy);
    L2_bias_RM(il) = norm(bias_RM);
    L2_bias_RF(il) = norm(bias_RF);
   
    a1 = (s-1)/(2*(N+2));
    a2 = (N + 2*s)/(2*(N+2));
    
    % FIM
    FIM_Sigma = K * D_n' * (a1*Inv_Shape_S(:)*Inv_Shape_S(:)' + a2*kron(Inv_Shape_S,Inv_Shape_S)) * D_n;
    % Constrained CRB
    CRB = U*inv(U'*FIM_Sigma*U)*U';
    
    % Semiparametric Efficient FIM
    SFIM_Sigma = K * D_n'  * a2*(kron(Inv_Shape_S,Inv_Shape_S) - (1/N)*Inv_Shape_S(:)*Inv_Shape_S(:)') * D_n;
    
    SCRB = U*inv(U'*SFIM_Sigma*U)*U';
    
    CR_Bound(il) = norm(CRB,'fro');
    SCR_Bound(il) = norm(SCRB,'fro');
end


color_matrix(1,:)=[0 0 1]; % Blue
color_matrix(2,:)=[1 0 0]; % Red
color_matrix(3,:)=[0 0.5 0]; % Dark Green
color_matrix(4,:)=[0 0 0]; % Black
color_matrix(5,:)=[0 0.5 1]; % Light Blue
color_matrix(6,:)=[1 0.3 0.6]; % Pink
color_matrix(7,:)=[0 0.9 0]; % Light Green

line_marker{1}='-s';
line_marker{2}='--d';
line_marker{3}=':^';
line_marker{4}='-.p';
line_marker{5}='-o';
line_marker{6}='--h';
line_marker{7}='-.*';

figure(1)
semilogy(svect,L2_bias_SCM,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:));
hold on
semilogy(svect,L2_bias_NTy,line_marker{4},'LineWidth',1,'Color',color_matrix(4,:),'MarkerEdgeColor',color_matrix(4,:),'MarkerFaceColor',color_matrix(4,:),'MarkerSize',8);
hold on
semilogy(svect,L2_bias_RM,line_marker{5},'LineWidth',1,'Color',color_matrix(5,:),'MarkerEdgeColor',color_matrix(5,:),'MarkerFaceColor',color_matrix(5,:),'MarkerSize',8);
hold on
semilogy(svect,L2_bias_RF,line_marker{6},'LineWidth',1,'Color',color_matrix(6,:),'MarkerEdgeColor',color_matrix(6,:),'MarkerFaceColor',color_matrix(6,:),'MarkerSize',8);
grid on
axis([0.1 2 0 0.1])
xlabel('Shape parameter: s');ylabel('Frobenius norm');
legend('SCM','Tyler','R_{vdW}','R_t')
title('Bias in Euclidean norm')


figure(2)
semilogy(svect,CR_Bound,line_marker{1},'LineWidth',1,'Color',color_matrix(1,:),'MarkerEdgeColor',color_matrix(1,:),'MarkerFaceColor',color_matrix(1,:));
hold on
semilogy(svect,SCR_Bound,line_marker{2},'LineWidth',1,'Color',color_matrix(2,:),'MarkerEdgeColor',color_matrix(2,:),'MarkerFaceColor',color_matrix(2,:));
hold on
semilogy(svect,Fro_MSE_SCM,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:));
hold on
semilogy(svect,Fro_MSE_NTy,line_marker{4},'LineWidth',1,'Color',color_matrix(4,:),'MarkerEdgeColor',color_matrix(4,:),'MarkerFaceColor',color_matrix(4,:),'MarkerSize',8);
hold on
semilogy(svect,Fro_MSE_RM,line_marker{5},'LineWidth',1,'Color',color_matrix(5,:),'MarkerEdgeColor',color_matrix(5,:),'MarkerFaceColor',color_matrix(5,:),'MarkerSize',8);
grid on
semilogy(svect,Fro_MSE_RF,line_marker{6},'LineWidth',1,'Color',color_matrix(6,:),'MarkerEdgeColor',color_matrix(6,:),'MarkerFaceColor',color_matrix(6,:),'MarkerSize',8);
axis([0.1 2 0.2 0.6])
xlabel('Shape parameter: s');ylabel('Frobenius norm');
legend('CCRB','CSCRB','SCM','Tyler','R_{vdW}','R_t')
title('MSE in Frobenus norm')
