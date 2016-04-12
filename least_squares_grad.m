function [g] = least_squares_grad(F, ytr)

n = length(F);
m = length(F{n-1});
Fnob = (F{n-1})(:,1:m-1)';
g = (F{n} - ytr)' * (F{n-1});