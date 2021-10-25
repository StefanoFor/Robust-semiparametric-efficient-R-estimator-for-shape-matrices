function J_phi = jacobian_constraint(N)

J_phi = zeros(N*(N+1)/2,1);

for j = 1:N;
    index_con(j) = 1 + N*(j-1) - (j-1)*(j-2)/2;
end

J_phi(index_con,1) = 1;
