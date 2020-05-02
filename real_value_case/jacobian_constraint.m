function J_phi = jacobian_constraint(N)

%% Real
J_phi = zeros(N*(N+1)/2,1);

for j = 1:N-1;
    index_con(j) = 1 + N*(j-1) - (j-1)*(j-2)/2;
end
% 
% J_phi(N*(N+1)/2,index_con) = -1;
% J_phi(N*(N+1)/2,N*(N+1)/2) = 0;

%% Complex
% J_phi = zeros(N^2,1);
% 
% for j = 1:N;
%     index_con(j) = j + N*(j-1);
% end

J_phi(index_con,1) = 1;
