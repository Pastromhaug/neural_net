function [G] = back_prop(W, F, yclass, ytr)

n = length(W);

G = cell(size(W));
num_y = size(yclass);
   
G{n} = least_squares_grad(F, ytr)';

for i = n-1:-1:1
    [n,m] = size(F{i+1});
    Fnob = F{i+1}(:,1:m-1);
    [gn,gm] = size(G{i+1});
    Gnob = G{i+1}(1:gn,1:gm-1);
    G{i} = relU_grad(Fnob, F{i}).*sum(Gnob,1);
endfor;



