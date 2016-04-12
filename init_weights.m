function [W] = init_weights(input_dim, output_dim, depth, width, w_mean, w_std)

W = cell(1,depth+1);
W{1} = normrnd(w_mean, w_std, input_dim+1, width);
for i = 2:depth
    W{i} = normrnd(w_mean, w_std, width+1, width); 
endfor;
W{depth+1} = normrnd(w_mean, w_std, width+1, output_dim);
    