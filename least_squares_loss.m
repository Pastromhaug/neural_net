function [l] = least_squares_loss(yclass, ytr)
    
l = sum(1/2*(ytr - yclass).^2);

    