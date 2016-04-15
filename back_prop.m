function [G] = back_prop(W, A, Z, ytr, loss_grad, non_lin_grad)


n = length(W);
G = cell(1,n);
dZ = cell(1,n);
dA = cell(1,n);
yclass = A{n};

dA{n} = loss_grad(A{n}, ytr);

for i = n:-1:2
    G{i} = dA{i}*(Z{i})';
    dZ{i} = (W{i})'*dA{i};
    [dzn,~] = size(Z{i});
    dZnob = dZ{i}(1:dzn-1,:);
    dA{i-1} = non_lin_grad(A{i-1}) .* dZnob;
end;
G{1} = dA{1}*(Z{1})';