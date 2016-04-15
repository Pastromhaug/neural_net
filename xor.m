function x = xor(x,y)

if round(x) == round(y),
    x = [1;0];
else,
    x = [0;1];
endif;