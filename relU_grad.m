function [g] = relU_grad(f, fprev)
    
n = length(f);
m = length(fprev); 
 
g = zeros(m,n);
    
for i = 1:size(f,2)
    if f(i) == 0
        g(:,i) = zeros(m,1);
    else,
        g(:,i) = fprev';
    endif;
    
endfor;