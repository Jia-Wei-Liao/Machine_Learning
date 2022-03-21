function [Y] = ReLU(X)
Y = X.*heaviside(X);

end