%%
% code adapted from Piotr Bojanowski

function [ A ] = build_norm( K, lambda )
%BUILD_NORM Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
  lambda  = 0.0001;
end
K = double(K);
[n,d] = size(K);
In = eye(n);
Pn = In - ones(n)/n;

B = Pn * K * Pn + n * lambda * In;

[V,D] = eig(B);
%fprintf('blah\n');
%keyboard;
A = 1000 * sqrt(lambda) * sqrtm(inv(D)) * V' * Pn;

end

%lambda * Pn' * B * Pn
