function [Xtr,Ytr] = init_training_data(num_params, output_dim, num_pts, f, min_f_param, max_f_param)

Xtr = rand(num_pts, num_params) * (max_f_param - min_f_param) + min_f_param;
Xtr = [Xtr, ones(num_pts,1)];  % for bias
Ytr = zeros(num_pts,output_dim);


for i = 1:num_pts
    vars = Xtr(i,1:num_params);
    
    if num_params == 1
        Ytr(i,:) = f(vars);
    elseif num_params == 2
        Ytr(i,:) = f(vars(1),vars(2));
    endif;
    
endfor;




