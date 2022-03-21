function [Z] = CrossEntropy(X, Y)
Z = sum(-Y.*log(X));

end