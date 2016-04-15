% init_training_data params
num_params = 2;
num_pts = 1000;
f = @(x,y) xor(x,y); 
min_f_param = 0;
max_f_param = 1;

% init_weights params
output_dim = 2;
depth = 3;
width = 100;
w_mean = 0;
w_std = 0.1;

% plot function f
%x = min_f_param:0.1:max_f_param;
%plot(x,arrayfun(f,x));

[Xtr,Ytr] = init_training_data(num_params, output_dim, num_pts, f, min_f_param, max_f_param);

[Winit] = init_weights(num_params, output_dim, depth, width, w_mean, w_std);
disp("training data and weights initialzied.");
