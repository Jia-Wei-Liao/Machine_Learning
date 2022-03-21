clc; clear; close all;

addpath('function/')

X  = [0.26; 0.33];                 % 2 x 1
Y  = [1; 0];                       % 2 x 1
lr = 0.5;

W1 = [0.1 0.5; 0.2 0.4; 0.4 0.1];  % 3 x 2
b1 = [0.3; 0.3; 0.3];              % 3 x 1
W2 = [0.1 0.5 0.3; 0.2 0.1 0.4];   % 2 x 3
b2 = [0.7; 0.7];                   % 2 x 1

for i = 1:1
    % Forward
    Y1 = W1*X + b1;                    % 3 x 1
    Z1 = ReLU(Y1);                     % 3 x 1
    Y2 = W2*Z1 + b2;                   % 2 x 1
    Z2 = Softmax(Y2);                  % 2 x 1
    L  = CrossEntropy(X, Y);

    % Backward
    dL_Z2  = -Y./Z2;                   % 2 x 1
    dZ2_Y2 = dSoftmax(Y2);             % 2 x 2
    dL_Y2  = dZ2_Y2 * dL_Z2;           % 2 x 1
    dZ1_Y1 = dReLU(Y1);                % 3 x 1

    W1 = W1 - lr * W2' * dL_Y2 * X';   % 3 x 2
    b1 = b1 - lr * W2' * dL_Y2;        % 3 x 1

    W2 = W2 - lr * dL_Y2 * Z1';        % 2 x 3
    b2 = b2 - lr * dL_Y2;              % 2 x 1

end
