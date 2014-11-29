%%
% brief: the face optimization part of the overal optimization
% input:
%   - train_data_info: the structure containing all the training data for the episode
%   - lambda: the regularization for the face optimization QP
%   - lambda_main_face: face optimization cost strength
% output:
%   - prob: the QP problem in mosek format

function prob = makeQP_face(train_data_info, lambda, lambda_main_face)

if isempty(lambda)
  lambda  = 0.001;
end


face_kernel = train_data_info.Kernel_face;


if ~isfield(train_data_info, 'faces_add_term')
train_data_info.faces_add_term = ...
      zeros(size(train_data_info.A_face,2), size(train_data_info.cast_to_bag_actor,1));
end


%% First add complementary information to speaker data
AA = train_data_info.track_to_bag_speaker;
BB = train_data_info.cast_to_bag_speaker;

%% Scene based constraints
AA = [AA train_data_info.track_to_bag_scene];
BB = [BB train_data_info.cast_to_bag_scene];

%to account for noise, extend AA by a small window
AA = extend_AA(AA, 3);

faces_add_term = train_data_info.faces_add_term;
isQP = false;

fprintf('making PSD ...\n');
tic;
face_kernel_psd = makePSD(face_kernel);
toc;

fprintf('making Bs ...\n');
tic;
Bs = build_norm(face_kernel_psd, lambda);
toc;

fprintf('Setting up problem ...\n');


tic;
  prob = get_mosek_prob(AA, BB, Bs, lambda_main_face, faces_add_term, isQP);
toc;

%keyboard;

function AA = extend_AA(AA,lim_val)

for i = 1:size(AA,2)
  if(sum(AA(:,i))<lim_val)
    max_id = max(find(AA(:,i)==1));
    num_max = lim_val-sum(AA(:,i));
    max_val = min(size(AA,1),max_id+num_max);
    if((max_id) < size(AA,1))
      AA((max_id+1):max_val,i) = 1;
    end
  end
end
