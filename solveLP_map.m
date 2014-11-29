%%
% brief: solve for the mapping matrix in the overall cost
%
% input:
%   - train_data_info: structure with training data information
%   - coref_data: strucutre with additional coref data information
%   - lambda_map: option to avoid a quadtratic term during optimization
%                 (not used in practice, since it slows things and gives no improvement)
%
% output:
%   - Q: the mapping matrix solved for

function Q = sovleLP_map(train_data_info, coref_data, lambda_map)


if (nargin < 3)
  lambda_map=[];
end

optimizerSetup;
M_clean = coref_data.alignment_scene_mask;

lin_weight   = train_data_info.map_add_term;

[I_ind, J_ind] = find(M_clean);
ctr = 1;
[N1, N2] = size(M_clean);
A_temp_const    = sparse(N1*N2, numel(I_ind));
A_simplex_const = sparse(N2, numel(I_ind));
fprintf('preparing temporal constraint\n');
tic;
for j = 1:(N2-1)
  inds_1 = find(J_ind==j);
  inds_2 = find(J_ind==(j+1));
  A_simplex_const(j,inds_1) = 1;
  I_inds_1 = I_ind(inds_1);
  I_inds_2 = I_ind(inds_2);
  min_i    = min(I_inds_2);
  max_i    = max(I_inds_1);
  for k = min_i:max_i
    add_ids_1 = inds_1(I_inds_1 <=k);
    add_ids_2 = inds_2(I_inds_2 <=k);
    A_temp_const(ctr, add_ids_1) = -1;
    A_temp_const(ctr, add_ids_2) = 1;
    ctr = ctr + 1;
  end
end
toc;

A_simplex_const(N2, J_ind==(N2)) = 1;

A_temp_const = A_temp_const(1:(ctr-1),:);
prob.a            = [A_temp_const; A_simplex_const];
prob.buc          = [zeros(ctr-1,1); ones(N2,1)];
prob.blc          = [-inf(ctr-1,1); ones(N2,1)];
prob.blx          = zeros(numel(I_ind),1);
prob.bux          = prob.blx + 1;
prob.c            = -lin_weight(find(M_clean));

% option for including a quadtratic term (does worse)
if ~isempty(lambda_map)
  Q          = lambda_map*eye(numel(I_ind));
  [prob.qosubi, prob.qosubj, prob.qoval] = find(Q);
end

[~, res]          = mosekopt('minimize', prob);
Q = zeros(N1,N2);
Q(find(M_clean)) = res.sol.itr.xx;

