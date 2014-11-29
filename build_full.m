%%
% code adpated from Piotr Bojanowski

function [ A ] = build_full( K, lambda )

if nargin < 2
  lambda  = 0.0001;
end

[n,d] = size(K);
In = eye(n);
Pn = In - ones(n)/n;

B = Pn * K * Pn + n * lambda * In;
A = lambda * Pn' * B * Pn;
