function [F] = forward_pass(W, x)

n = length(W);
F = cell(i, n+1);

F(1,1) = x;

for i= 1:n-1
    f = F{1,i}*W{i};
    F(1,i+1) = [relU(f,0),1];
endfor;

f = F{1,n}*W{n};
F(1,n+1) = f;



function o = relU(x)
    o = max(x,0);
endfunction;
endfunction;