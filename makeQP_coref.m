%%
% brief: setup the QP optimization problem in mosek format for the coref problem

% input:
%   - pair_features: fatures between pairs of mentions in our script, every cell corresponds to a pair
%   - pair_ids: a cell structure which gives the indices of the pair of mentions in the pair_features part
%   - gcast: a pre-computed utility matrix which gives the edit-distance between a mention word and the different
%            "cast names" in our script
%   - A_ment: a mapping matrix which maps the mentions which go into the optimization from all mentions in the script
%   - coref_add_term: term which contains mapping and face information updated by the Update... code
%   - lambda: the strength of the coref cost term in overall cost
%   - lambda_norm: regualrization parameter in QP of coref
%   - slack_eps2: the slack value for QP
%
% output:
%   - prob_QP: the problem in mosek QP format
%   - animate_mentions: only animate mentions are used in optimization (the list of those mentions)
%   - zinds, ainds, pair_ids: intermediated ids for debugging (to be deprecated)

function [prob_QP, animate_mentions, zinds, ainds, pair_ids] = ...
                          makeQP_coref(pair_features, pair_ids, gcast, A_ment,...
                          coref_add_term, ...
                          lambda, lambda_norm, slack_eps2)

eval_ments = zeros(size(A_ment,1),1);

for i = 1:size(A_ment,1)
  if sum(A_ment(i,:))>0
    eval_ments(i) = find(A_ment(i,:));
  end
end
eval_ments = eval_ments(eval_ments > 0);

[pair_features, pair_ids, gcast, animate_mentions] = retainSubset(pair_features, ...
                                                  pair_ids, gcast, eval_ments);

gname_orig = zeros(size(gcast,1),1) + size(gcast,2);

for i = 1:size(gcast,1)
  [maxg, maxid] = max(gcast(i,:));
  if maxg>=0.75
    gname_orig(i) = maxid;
  end
end


%keyboard;

all_pairfeat = cell2mat(pair_features);
all_pairids  = cell2mat(pair_ids);
pair_orig    = zeros(size(all_pairids));


size_Z              = size(gcast');
size_A              = [size(all_pairids,1) 1];
num_cast            = size(gcast,2);
size_out            = size_Z(1)*size_Z(2) + size_A(1);

ainds = (1:size_A(1));
zinds = (size_A(1)+1):(size_A(1)+ size_Z(1)*size_Z(2));

pair_kernel = all_pairfeat*all_pairfeat';

max_norm_constraint = sparse(num_cast*2*size_A(1), size_out);
simplex_constraint  = sparse(numel(pair_ids) + size_Z(2), size_out);

ctr_beg = 1;

ctr_beg = 1;
for i = 1:numel(pair_ids)
  ctr_end = ctr_beg - 1 + numel(pair_ids{i});
  pair_orig(ctr_beg:ctr_end) = i;
  simplex_constraint(i, ctr_beg:ctr_end) = 1;
  ctr_beg = ctr_end + 1;
end

%keyboard;
for i = 1:size_A(1)

  if (mod(i,100)==0)
    fprintf('%d/%d \n', i, size_A(1));
  end


  %if(i==198)
  %  keyboard;
  %end

  % set to the gname_orig if connected to self
  if(pair_orig(i) == all_pairids(i,1))
    beg_id = (i-1)*(num_cast*2) + 1;
    max_norm_constraint(beg_id,i) = 1;
    g_orig_id = size_A(1) + (pair_orig(i)-1)*num_cast + gname_orig(pair_orig(i));
    max_norm_constraint(beg_id, g_orig_id) = -1;
    continue;
  end

  beg_id = (i-1)*(num_cast*2) + 1;
  end_id1 = (2*i - 1)*num_cast;
  end_id2 = i*(num_cast*2);
  
  max_norm_constraint(beg_id:end_id2, i) = 1;

  id1_j = size_A(1) + [((pair_orig(i)-1)*num_cast + 1), pair_orig(i)*num_cast];
  id2_j = size_A(1) + [((all_pairids(i,1)-1)*num_cast + 1), all_pairids(i,1)*num_cast];

  max_norm_constraint(beg_id:end_id1, id1_j(1):id1_j(2)) = eye(num_cast);
  max_norm_constraint(beg_id:end_id1, id2_j(1):id2_j(2)) = -eye(num_cast);

  max_norm_constraint((end_id1+1):end_id2, id1_j(1):id1_j(2)) = -eye(num_cast);
  max_norm_constraint((end_id1+1):end_id2, id2_j(1):id2_j(2)) = eye(num_cast);

%  if (pair_orig(i)==433)
%    keyboard;
%  end

end



max_norm_uc = ones(num_cast*2*size_A(1), 1);
max_norm_lc = -ones(num_cast*2*size_A(1), 1);


valid_constraints = sum(abs(max_norm_constraint),2);
%keyboard;
% if connected to self, then set mention to gorig_name
max_norm_uc(valid_constraints==2) = 0;

max_norm_constraint = max_norm_constraint(valid_constraints>=1, :);
max_norm_uc         = max_norm_uc(valid_constraints>=1);
max_norm_lc         = max_norm_lc(valid_constraints>=1);

for i = 1:size_Z(2)
  ctr_beg = size_A(1) + (i-1)*num_cast + 1;
  ctr_end = ctr_beg + num_cast - 1;
  simplex_constraint(i + numel(pair_ids), ctr_beg:ctr_end) = 1;
end

simplex_eq_values = ones(numel(pair_ids) + size_Z(2), 1);

%keyboard;

Qa_eq_values = zeros( size_A(1),1)-1;
Qa_eq_values_islack = zeros(size_A(1),1);

for i = 1:size_A(1)

  % if pronoun, do not link to itslef
  if (all_pairfeat(i,50)==1 && (numel(pair_ids{pair_orig(i)})>1))
    Qa_eq_values(i) = 0;
    continue;
  end

  % if the word is the same as a cast-name, then link to itself
  if (all_pairfeat(i,end) >= 0.95)
    Qa_eq_values(i) = 1;
    continue;
  end
  
  if (all_pairfeat(i,50) == 0 && all_pairids(i) == pair_orig(i))
    Qa_eq_values(i) = 1;
    Qa_eq_values_islack(i) = 1;
    continue;
  end

  % if genders don't match, don't link
  if (all_pairfeat(i,37) ~= 1 && all_pairids(i) ~= pair_orig(i))
    Qa_eq_values(i) = 0;
    Qa_eq_values_islack(i) = 1;
  end
end

qa_fixids = find(Qa_eq_values >= 0);
Qa_eq_values = Qa_eq_values(qa_fixids);
Qa_eq_values_islack = Qa_eq_values_islack(qa_fixids);

Qa_constraints = sparse(numel(qa_fixids), size_out);
Qa_ids = sub2ind(size(Qa_constraints), 1:size(Qa_constraints,1), qa_fixids');
Qa_constraints(Qa_ids) = 1;

Qz_eq_values = zeros(size_Z(2), 1);
[maxg, maxid] = max(gcast');
for i = 1:size_Z(2)
  if (maxg(i)==1 && maxid(i) < num_cast)
    Qz_eq_values(i) = (i-1)*num_cast + maxid(i) + size_A(1);
  end
end

qz_fixids = find(Qz_eq_values > 0);
Qz_eq_values = Qz_eq_values(qz_fixids);
Qz_constraints = sparse(numel(qz_fixids), size_out);
Qz_ids = sub2ind(size(Qz_constraints), 1:size(Qz_constraints, 1), Qz_eq_values');
Qz_constraints(Qz_ids) = 1;
Qz_eq_values = ones(numel(qz_fixids), 1);

A_kernel = build_full(pair_kernel, lambda_norm);

%A_kernel = A_kernel'*A_kernel;
Bs = double(A_kernel);
[ij, ik, val] = find(tril(Bs));

prob.qosubi = ij;
prob.qosubj = ik;
prob.qoval  = val;


prob.blc = [Qa_eq_values; Qz_eq_values; max_norm_lc; simplex_eq_values];
prob.buc = [Qa_eq_values; Qz_eq_values; max_norm_uc; simplex_eq_values];

%% Adding slack constraints
%keyboard;
num_slacks = sum(Qa_eq_values_islack>=1);
qa_slack_vals = Qa_eq_values_islack(Qa_eq_values_islack>=1);
slack_eps = 0.01;
slack_eps_vals = ones(num_slacks,1)*sqrt(slack_eps);
slack_eps_vals(qa_slack_vals >=2) = sqrt(slack_eps2);

Qa_slack_constraint = zeros(size(Qa_constraints, 1), num_slacks);
Qa_slack_ids = sub2ind(size(Qa_slack_constraint), find(Qa_eq_values_islack'>=1), 1:num_slacks);
Qa_slack_constraint(Qa_slack_ids) = -1;

prob.qosubi = [prob.qosubi; size_out+(1:num_slacks)'];
prob.qosubj = [prob.qosubj; size_out+(1:num_slacks)'];
prob.qoval  = [prob.qoval;  slack_eps_vals];

Qa_constraints = [Qa_constraints Qa_slack_constraint];
Qz_constraints = [Qz_constraints sparse(size(Qz_constraints,1), num_slacks)];
max_norm_constraint = [max_norm_constraint sparse(size(max_norm_constraint, 1), num_slacks)];
simplex_constraint  = [simplex_constraint  sparse(size(simplex_constraint, 1), num_slacks)];

prob.a   = [Qa_constraints; Qz_constraints; max_norm_constraint; simplex_constraint];
prob.bux = ones(size_out, 1);
prob.blx = zeros(size_out, 1);

prob.bux = [prob.bux; inf(num_slacks,1)];
prob.blx = [prob.blx; -inf(num_slacks,1)];

prob.c   = zeros(numel(prob.bux),1);
%keyboard;
coref_add_term = coref_add_term(:, animate_mentions);
prob.c(zinds) = -lambda*coref_add_term(:)';

prob_QP = prob;


function [pair_features_new, pair_ids_new, gcast_new, animate_mentions] = retainSubset(pair_features, pair_ids, gcast, eval_ments)

% only retain those mentions which are proper nouns or pronouns

for i = 1:numel(pair_features)
  animval(i) = pair_features{i}(end, 43);
  gcastval(i) = pair_features{i}(end, end);
end

valid_mentions = find((animval==1) | (gcastval > 0.90));
valid_mentions = union(valid_mentions, eval_ments);
valid_mentions = sort(unique(valid_mentions), 'ascend');

isvalid = zeros(1, numel(pair_ids));
pair_features_new = cell(numel(pair_ids), 1);
pair_ids_new      = cell(numel(pair_ids), 1);
for i = 1:numel(pair_ids)
  [pair_ids_new{i}, int_ids, ~] = intersect(pair_ids{i}, valid_mentions);
  pair_ids_new{i} = pair_ids_new{i}';
  pair_features_new{i} = pair_features{i}(int_ids, :);
  isvalid(i) = any(valid_mentions==i);
end

pair_features_new = pair_features_new(isvalid > 0);
pair_ids_new      = pair_ids_new(isvalid > 0);
gcast_new         = gcast(isvalid>0, :);

animate_mentions = find(isvalid>0);

for i = 1:numel(pair_ids_new)
  %keyboard;
  try
    pair_ids_new{i} = mapFromB2A(animate_mentions, pair_ids_new{i}');
  catch
    pair_ids_new{i} = mapFromB2A(animate_mentions, pair_ids_new{i});
  end
  val_id = find(pair_ids_new{i}==i);
  non_val_id = [1:(val_id-1) (val_id+1):numel(pair_ids_new{i})];
  if ~isempty(val_id)
    pair_ids_new{i}      = pair_ids_new{i}([non_val_id val_id]);
    pair_features_new{i} = pair_features_new{i}([non_val_id val_id],:);
  else
    fprintf('something wrong here!\n');
    error('not a valid antecedent list !!!\n');
  end

  %try
  %  assert(pair_ids_new{i}(end)==i);
  %catch
  %  keyboard;
  %end
end

