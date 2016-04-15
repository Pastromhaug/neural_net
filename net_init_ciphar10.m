% init_training_data params
%num_params = 3072;
%num_pts = 1000;
%f = @(x,y) xor(x,y); 
%min_f_param = 0;
%max_f_param = 1;

% init_weights params
%output_dim = 1;
depth = 3;
width = 100;
w_mean = 0;
w_std = 0.1;

% plot function f
%x = min_f_param:0.1:max_f_param;
%plot(x,arrayfun(f,x));

load data_batch_1.mat;
Xtr = double(data');
Ytr = cell2mat(arrayfun(@(x) eye(10)(:,x+1), double(labels'), "UniformOutput",false));
[num_params,num_pts] = size(Xtr);
[output_dim,~] = size(Ytr);

[Winit] = init_weights(num_params, output_dim, depth, width, w_mean, w_std);
disp("training data and weights initialzied.");
