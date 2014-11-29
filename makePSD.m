%%
% code adapted from Piotr Bojanowski

function bigK = makePSD(bigK)

bigK = (bigK+bigK')/2;
[V,D] = eig(bigK);
bigK = V * abs(D) * V';
bigK = (bigK+bigK')/2;

end
