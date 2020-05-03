clear all
close all
clc


Ns=10^6; % monte carlo trials
Max_it = 250;
N = 8;
perturbation_par = 0.01;
nu_par = 3;

ro=0.8*exp(1j*2*pi/5);
sigma2 = 4;

lambdavect=1.1:1:20.1;

Nl=length(lambdavect);

K = 5*N;
n=[0:N-1];

rx=ro.^n; % Autocorrelation function
Sigma = toeplitz(rx);

L=chol(Sigma);
L=L';

Shape_S = N*Sigma/trace(Sigma);
Inv_Shape_S = inv(Shape_S);
theta_true = Shape_S(:);

I_N = eye(N);
J_phi = I_N(:);
U = null(J_phi');

DIM = N^2;

tic
for il=1:Nl
    
    lambda = lambdavect(il)
    eta = lambda/(sigma2*(lambda-1));
    scale=eta/lambda;
    
    MSE_SCM = zeros(DIM,DIM);
    MSE_NFP = zeros(DIM,DIM);
    MSE_RM = zeros(DIM,DIM);
    bias_SCM = zeros(DIM,1);
    bias_NFP = zeros(DIM,1);
    bias_RM = zeros(DIM,1);
    
    parfor ins=1:Ns
        
        w = (randn(N,K)+1j.*randn(N,K))/sqrt(2);
        R = gamrnd(lambda,scale,1,K);
        x = L*w;
        y = sqrt(1./(repmat(R,N,1))).*x;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SCM
        SCM = y*y'/K;
        Scatter_SCM = N*SCM/trace(SCM);
        err_v = Scatter_SCM(:)-theta_true;
        bias_SCM = bias_SCM + err_v/Ns;
        err_MAT = err_v(1:end,:)*err_v(1:end,:)';
        MSE_SCM = MSE_SCM + err_MAT/Ns;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Tyler matrix estimator
        FP = Tyler_est_11( y, Max_it);
        NFP = N*FP/trace(FP);
        err_v = NFP(:)-theta_true;
        bias_NFP = bias_NFP + err_v/Ns;
        err_NFP = err_v(1:end,:)*err_v(1:end,:)';
        MSE_NFP= MSE_NFP + err_NFP/Ns;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % R - estimator
        [RM, a] = R_G_est( y, FP, nu_par, perturbation_par);
        RM = N*RM/trace(RM);
        err_v = RM(:)-theta_true;
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
    
    a1 = -1/(N+lambda+1);
    a2 = (lambda + N)/(N+lambda+1);
    
    % FIM
    FIM_Sigma = K * (a1*Inv_Shape_S(:)*Inv_Shape_S(:)' + a2*kron(Inv_Shape_S.',Inv_Shape_S));
    
    CRB = U*inv(U'*FIM_Sigma*U)*U';
    
    SFIM_Sigma = K * a2*(kron(Inv_Shape_S.',Inv_Shape_S) - (1/N)*Inv_Shape_S(:)*Inv_Shape_S(:)');
    
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
semilogy(lambdavect,L2_bias_SCM,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:));
hold on
semilogy(lambdavect,L2_bias_NFP,line_marker{4},'LineWidth',1,'Color',color_matrix(4,:),'MarkerEdgeColor',color_matrix(4,:),'MarkerFaceColor',color_matrix(4,:),'MarkerSize',8);
grid on;
semilogy(lambdavect,L2_bias_RM,line_marker{5},'LineWidth',1,'Color',color_matrix(5,:),'MarkerEdgeColor',color_matrix(5,:),'MarkerFaceColor',color_matrix(5,:),'MarkerSize',8);
xlabel('Shape parameter: \lambda');ylabel('Frobenius norm');
legend('CSCM','C-Tyler','R-est')
title('Bias in Euclidean norm')


figure(2)
semilogy(lambdavect,CR_Bound,line_marker{1},'LineWidth',1,'Color',color_matrix(1,:),'MarkerEdgeColor',color_matrix(1,:),'MarkerFaceColor',color_matrix(1,:));
hold on
semilogy(lambdavect,SCR_Bound,line_marker{2},'LineWidth',1,'Color',color_matrix(2,:),'MarkerEdgeColor',color_matrix(2,:),'MarkerFaceColor',color_matrix(2,:));
hold on
semilogy(lambdavect,Fro_MSE_SCM,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:));
hold on
semilogy(lambdavect,Fro_MSE_NFP,line_marker{4},'LineWidth',1,'Color',color_matrix(4,:),'MarkerEdgeColor',color_matrix(4,:),'MarkerFaceColor',color_matrix(4,:),'MarkerSize',8);
grid on;
semilogy(lambdavect,Fro_MSE_RM,line_marker{5},'LineWidth',1,'Color',color_matrix(5,:),'MarkerEdgeColor',color_matrix(5,:),'MarkerFaceColor',color_matrix(5,:),'MarkerSize',8);
axis([1 20.1 0.3 0.6])
xlabel('Shape parameter: \lambda');ylabel('Frobenius norm');
legend('CCRB','CSCRB','CSCM','C-Tyler','R-est')
title('MSE in Frobenus norm')



