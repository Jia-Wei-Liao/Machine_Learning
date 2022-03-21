function [Y] = Softmax(X)
e = exp(1);
Y = e.^X / sum(e.^X);

end