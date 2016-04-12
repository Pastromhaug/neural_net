% rest of params
epochs = 200;
learning_rate = 1e-2;

W = Winit;
% training neural net
Yclass = Ytr;
YclassRaw = Ytr;
Loss = Ytr;
EpochLosses = zeros(epochs);

plot_colors = zeros(num_pts,3);

for i = 1:epochs
    epoch_loss = 0;
    numright = 0;
    accuracy = 0;
    for j = 1:num_pts
        xtr = Xtr(j,:);
        ytr = Ytr(j,:);
        F = forward_pass(W, xtr);
        yclass = F{length(W)+1};
        loss = least_squares_loss(yclass, ytr);
        
        G = back_prop(W, F, yclass, ytr);
        
        % update weights
        for k = 1:size(W,2)
            W{k} = W{k} - learning_rate*G{k};
        endfor;
               
        
        % stats
        epoch_loss = epoch_loss + loss;
        if sign(yclass(1) - yclass(2)) == sign(ytr(1) - ytr(2)) ,
            numright = numright + 1;
        endif;
        if isequal(round(yclass),[1,0]) ,
            plot_colors(j,:) = [250,0,0];
        else,
            plot_colors(j,:) = [0,250,0];
        endif;
        %Yclass(j) = sign(yclass);
        %YclassRaw(j) = yclass;
        %Loss(j) = loss;
        
    endfor;
    epoch_loss
    EpochLosses(i) = epoch_loss;
    figure(1);
    s = scatter(Xtr(:,1),Xtr(:,2),s=8,c=plot_colors,"filled");
    figure(2);
    e = plot(EpochLosses);
    
    fflush(stdout);
    
    %accuracy = numright / num_pts
    %g_norm = sum(cellfun(@(x) sum(sum(x)), G))
    %[YclassRaw, Yclass, Ytr, Loss]
endfor;