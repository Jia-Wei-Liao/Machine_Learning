function [Z] = dSoftmax(X)
n = length(X);
Y = Softmax(X);
i = 1:n;
j = 1:n;
[J, I] = meshgrid(j, i);
Z =  diag(Y.*(1-Y)) + (eye(n) - ones(n)) .* (Y(I) .* Y(J));

end