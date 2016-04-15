% rest of params
epochs = 100;
learning_rate = 1e-2;
non_lin = @(x) max(x,0);
non_lin_grad = @(a) (a > 0);
loss_func = @(yclass, ytr) sum(1/2*(ytr - yclass).^2);
loss_grad = @(yclass, ytr) yclass - ytr;
batch_size = 1;
graph_loss = true; 
graph_accuracy = true;
graph_xor_colors = true;

W = Winit;
num_batches = ceil(num_pts/batch_size)

if graph_loss,
    EpochLosses = zeros(epochs);
end;
if graph_accuracy,
    Accuracies = zeros(epochs);
end;


for i = 1:epochs

%------ stat variables ---------------------
    if graph_loss,
        epoch_loss = 0;
    end;
    
    if graph_accuracy,
        numright = 0;
        accuracy = 0;
    end;
    
    if graph_xor_colors,
        plot_colors = zeros(num_pts,3);
    end;
    
%------ actual training --------------------
    for j = 1:num_batches
        indeces = (j-1)*batch_size+1 : 1 : mod(j*batch_size,num_pts+1);
        xtr = Xtr(:,j);
        ytr = Ytr(:,j);
        [A,Z] = forward_pass(W, xtr, non_lin);
        G = back_prop(W, A, Z, ytr, loss_grad, non_lin_grad);
        for k = 1:length(W);
            W{k} = W{k} - learning_rate*G{k};
        end;
               
%------- stats ----------------------------
        yclass = A{length(A)};
        if graph_loss,
            loss = sum(loss_func(yclass, ytr));
            epoch_loss = epoch_loss + loss;
        end;
        if graph_accuracy,
            [~,ix] = max(yclass);
            [~,ixtr] = max(ytr);
            for l = 1:length(ix)
                if ix(l) == ixtr(l),
                    numright = numright + 1;
                end;
            end;
        end;
        if graph_xor_colors,
            [~,yclasslen] = size(yclass);  
            for l = 1:yclasslen
                if isequal(round(yclass(:,l)),[1;0]) ,
                    plot_colors(j+l-1,:) = [250,0,0];
                else,
                    plot_colors(j+l-1,:) = [0,250,0];
                end;
            end;
        end;        
    end;

%------ plotting & printing to console-------
    if graph_xor_colors,
        figure(1);
        s = scatter(Xtr(1,:),Xtr(2,:),s=8,c=plot_colors,"filled");
        axis("square");
        fflush(stdout);
    end;    
    
    if graph_loss,
        EpochLosses(i) = epoch_loss;
        figure(2);
        e = plot(EpochLosses(1:i));
        fflush(stdout);
    end;
    
    if graph_accuracy,
        accuracy = numright / num_pts;
        Accuracies(i) = accuracy;
        figure(3);
        a = plot(Accuracies(1:i));
        fflush(stdout);
    end;    
end;