function [A,Z] = forward_pass(W, x, non_lin)

n = length(W);
A = cell(1,n);
Z = cell(1,n);

for i= 1:n
    Z{i} = [x;ones(1,size(x,2))];
    A{i} = W{i}*Z{i};
    x = non_lin(A{i});
end
