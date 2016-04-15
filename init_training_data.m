function [Xtr,Ytr] = init_training_data(num_params, output_dim, num_pts, f, min_f_param, max_f_param)

Xtr = rand(num_params,num_pts) * (max_f_param - min_f_param) + min_f_param;
Ytr = zeros(output_dim,num_pts);


for i = 1:num_pts
    vars = Xtr(1:num_params,i);
    
    if num_params == 1
        Ytr(:,i) = f(vars);
    elseif num_params == 2
        Ytr(:,i) = f(vars(1),vars(2));
    endif;
    
endfor;




