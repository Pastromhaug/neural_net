function [W] = init_weights(input_dim, output_dim, depth, width, w_mean, w_std)

W = cell(1,depth+1);
W{1} = normrnd(w_mean, w_std, width, input_dim+1);
for i = 2:depth
    W{i} = normrnd(w_mean, w_std, width, width+1); 
endfor;
W{depth+1} = normrnd(w_mean, w_std, output_dim, width+1);
    