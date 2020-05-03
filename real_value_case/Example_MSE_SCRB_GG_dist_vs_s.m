clear all
close all
clc


Ns=10^6; % monte carlo trials
N = 8;
Max_it = 250;

ro=0.8;
sigma2 = 4;

perturbation_par = 0.01;

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
    MSE_NFP = zeros(DIM,DIM);
    MSE_RM = zeros(DIM,DIM);
    bias_SCM = zeros(DIM,1);
    bias_NFP = zeros(DIM,1);
    bias_RM = zeros(DIM,1);
    a_mean = 0;
    
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
        FP = Tyler_Const_11( y, Max_it);
        NFP = N*FP/trace(FP);
        err_v = L_n*NFP(:)-theta_true;
        bias_NFP = bias_NFP + err_v/Ns;
        err_NFP = err_v(1:end,:)*err_v(1:end,:)';
        MSE_NFP= MSE_NFP + err_NFP/Ns;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % R - estimator
        [RM, a] = R_VDW_estimator( y, FP, perturbation_par);
        RM = N*RM/trace(RM);
        err_v = L_n*RM(:)-theta_true;
        bias_RM = bias_RM + err_v/Ns;
        err_RM = err_v(1:end,:)*err_v(1:end,:)';
        MSE_RM= MSE_RM + err_RM/Ns;
        
    end
    Fro_MSE_SCM(il) = norm(MSE_SCM,'fro');
    Fro_MSE_NFP(il) = norm(MSE_NFP,'fro');
    Fro_MSE_RM(il) = norm(MSE_RM,'fro');
    L2_bias_SCM(il) = norm(bias_SCM);
    L2_bias_NFP(il) = norm(bias_NFP);
    L2_bias_RM(il) = norm(bias_RM);
   
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
color_matrix(5,:)=[0.8 0.8 0]; % Yellow

line_marker{1}='-s';
line_marker{2}='--d';
line_marker{3}=':^';
line_marker{4}='-.p';
line_marker{5}='-.*';


figure(1)
semilogy(svect,L2_bias_SCM,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:));
hold on
semilogy(svect,L2_bias_NFP,line_marker{4},'LineWidth',1,'Color',color_matrix(4,:),'MarkerEdgeColor',color_matrix(4,:),'MarkerFaceColor',color_matrix(4,:),'MarkerSize',8);
grid on;
semilogy(svect,L2_bias_RM,line_marker{5},'LineWidth',1,'Color',color_matrix(5,:),'MarkerEdgeColor',color_matrix(5,:),'MarkerFaceColor',color_matrix(5,:),'MarkerSize',8);
xlabel('Shape parameter: s');ylabel('Frobenius norm');
legend('CSCM','C-Tyler','R-est')
title('Bias in Euclidean norm')


figure(2)
semilogy(svect,CR_Bound,line_marker{1},'LineWidth',1,'Color',color_matrix(1,:),'MarkerEdgeColor',color_matrix(1,:),'MarkerFaceColor',color_matrix(1,:));
hold on
semilogy(svect,SCR_Bound,line_marker{2},'LineWidth',1,'Color',color_matrix(2,:),'MarkerEdgeColor',color_matrix(2,:),'MarkerFaceColor',color_matrix(2,:));
hold on
semilogy(svect,Fro_MSE_SCM,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:));
hold on
semilogy(svect,Fro_MSE_NFP,line_marker{4},'LineWidth',1,'Color',color_matrix(4,:),'MarkerEdgeColor',color_matrix(4,:),'MarkerFaceColor',color_matrix(4,:),'MarkerSize',8);
grid on;
semilogy(svect,Fro_MSE_RM,line_marker{5},'LineWidth',1,'Color',color_matrix(5,:),'MarkerEdgeColor',color_matrix(5,:),'MarkerFaceColor',color_matrix(5,:),'MarkerSize',8);
axis([0.1 2 0.25 0.6])
xlabel('Shape parameter: s');ylabel('Frobenius norm');
legend('CCRB','CSCRB','CSCM','C-Tyler','R-est')
title('MSE in Frobenus norm')
