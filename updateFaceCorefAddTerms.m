%%
% brief: update the different $_add_term parts of the train_data_info. This is how the
%        the three parts of the cost function talk to each other.
% input:
%   - Y: the face results matrix
%   - Q: the mapping matrix
%   - Z: the coreference results matrix
%   - train_data_info: the strucutre with all training data
%   - coref_data: the structure with additional data for coref optimization
%
% output:
%   - train_data_info: updated train_data_info structure

function train_data_info = updateFaceCorefAddTerms(Y, Q, Z, train_data_info, coref_data)

num_cast = size(coref_data.gcast, 2);

train_data_info.coref_add_term = getCorefAddTerm(Y,Q, train_data_info.A_face,...
                              num_cast, coref_data.unary_unique, train_data_info.Za_mentids);
train_data_info.faces_add_term = getFacesAddTerm(Q,Z, ...
                          train_data_info.A_face, train_data_info.A_ment);
train_data_info.map_add_term   = getMapAddTerm  (Z,Y,...
                                train_data_info.A_face, train_data_info.A_ment);



%   ------------------------------------------
% |        Term to add to the coref_cost       |
%   ------------------------------------------
function coref_add_term = getCorefAddTerm(Y, Q, A_face, num_cast, unary_unique, Za_mentids)
%num_cast = numel(full_data.train_data_info.unique_names) + 1;
coref_add_term = zeros(num_cast , numel(unary_unique));
Z_from_face = Y*A_face'*Q;
for i = 1:size(Q,2)
  zam = find(unary_unique==Za_mentids(i));
  if ~isempty(zam)
    coref_add_term(:, zam) = coref_add_term(:, zam) + Z_from_face(:, i);
  end
end

%%% Term to add to the faces cost
function faces_add_term = getFacesAddTerm(Q, Z, A_face, A_ment)
faces_add_term  = A_face'*Q*A_ment*Z';

%%% Term to add to the mapping cost
function map_add_term   = getMapAddTerm(Z, Y, A_face, A_ment)
map_add_term = (A_ment*Z'*Y*A_face')';

