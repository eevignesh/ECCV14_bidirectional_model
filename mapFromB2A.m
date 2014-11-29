%%
% @brief: a(c) = b

function c = mapFromB2A(a, b)

[d, c] = find(bsxfun(@eq, b', a));
[~, idx] = sort(d, 'ascend');
c = c(idx);

%keyboard;
assert(all(a(c)==b));
