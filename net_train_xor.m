% rest of params
epochs = 10;
learning_rate = 1e-2;
non_lin = @(x) max(x,0);
non_lin_grad = @(a) (a > 0);
loss_func = @(yclass, ytr) sum(1/2*(ytr - yclass).^2);
loss_grad = @(yclass, ytr) yclass - ytr;
batch_size = 11;

W = Winit;
num_batches = ceil(num_pts/batch_size);
% training neural net
EpochLosses = zeros(epochs);
Accuracies = zeros(epochs);



for i = 1:epochs

%------ stat variables ---------------------
    epoch_loss = 0;
    numright = 0;
    accuracy = 0;
    plot_colors = zeros(num_pts,3);
    
%------ actual training --------------------
    for j = 1:num_batches
        batch_start = (j-1)*batch_size+1;
        batch_end = min(j*batch_size,num_pts);
        idx = batch_start:1:batch_end;
        xtr = Xtr(:,idx);
        ytr = Ytr(:,idx);
        [A,Z] = forward_pass(W, xtr, non_lin);
        G = back_prop(W, A, Z, ytr, loss_grad, non_lin_grad);
        for k = 1:length(W);
            W{k} = W{k} - learning_rate*G{k};
        end;
               
%------- stats ----------------------------
        yclass = A{length(A)};
        loss = sum(loss_func(yclass, ytr));
        epoch_loss = epoch_loss + loss;
        [x,ix] = max(yclass);
        [xtr,ixtr] = max(ytr);
        for l = 1:size(yclass,2)
            if ix(l) == ixtr(l),
                numright = numright + 1;
            endif;
            if isequal(round(yclass(:,l)),[1;0]) ,
                plot_colors((j-1)*batch_size+l,:) = [250,0,0];
            else,
                plot_colors((j-1)*batch_size+l,:) = [0,250,0];
            endif;
        end;        
    end;

%------ plotting & printing to console-------
    epoch_loss
    EpochLosses(i) = epoch_loss;
    accuracy = numright / num_pts;
    Accuracies(i) = accuracy;
    
    %plot_colors
    figure(1);
    s = scatter(Xtr(1,:),Xtr(2,:),s=8,c=plot_colors,"filled");
    axis("square");
    
    figure(2);
    e = plot(EpochLosses(1:i));
    
    figure(3);
    a = plot(Accuracies(1:i));
    
    fflush(stdout);
end;