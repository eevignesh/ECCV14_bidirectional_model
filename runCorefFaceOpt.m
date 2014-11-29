%%
% brief: Main function to be called to run the bidirectional optimization
%
% input:
%   - episdoe_name: name of the episode
%   - lambda_main_face: the strength of the face optimization cost in the overall cost (optimum value = 10)
%   - lambda_main_coref: the strength of the coref optimization cost in the overall cost (optimum value = 0.05)
%   - data_dir: main directory containing all the data (set to the path of 'datasets')
%   - slack_eps2: the slack parameter for QP optimization (use default value)
%   - lambda_coref:  regularization parameter for the coref QP
%   - lambda_face_1: regularization parameter for the face QP

function runCorefFaceOpt(episode_name, lambda_main_face, lambda_main_coref, data_dir, ...
            slack_eps2, lambda_coref, lambda_face_1)

optimizerSetup;

if nargin < 4
  data_dir = '/scail/scratch/u/vigneshr/joint_nlp_vision/datasets/';
end

%default parameters
if nargin < 7
  lambda_face_1 = 0.0001;
end

if nargin < 6
  lambda_coref  = 0.01;
end

if nargin < 5
  slack_eps2 = 1000;
end

isGoldCoref = 1;
episode_dir = [data_dir '/' episode_name '/'];

% load data files
final_data_release_dir             = [episode_dir '/data_release/'];
if ~exist(final_data_release_dir)
  unix(['mkdir ' final_data_release_dir]);
end
save_file_name = [final_data_release_dir '/bidirectional_data.mat'];
load(save_file_name);

% final results with map files
final_map_dir             = [episode_dir '/bidirectional_results/'];

if ~exist(final_map_dir)
  unix(['mkdir ' final_map_dir]);
end
% The linear terms corresponding to joint optimization
train_data_info = updateFaceCorefAddTerms(train_data_info.Y_init,...
                                            train_data_info.Q_init,...
                                            train_data_info.Z_init,...
                                            train_data_info,...
                                            coref_data);
train_data_info.Q_init = solveLP_map(train_data_info, coref_data);

% START ALTERNATING OPTMIZATION
toy_Y_file                = [final_map_dir sprintf('faces_Y_%f_%f_lf1%f', ...
                            lambda_main_coref, lambda_main_face, ...
                            lambda_face_1) '_face1_iter%02d.mat'];
toy_Z_file                = [final_map_dir sprintf('coref_frank_vnew_Z_%f_%f_lc%f_se%f', ...
                            lambda_main_coref, lambda_main_face, ...
                            lambda_coref, slack_eps2) '_face1_iter%02d.mat'];
toy_result_file           = [final_map_dir sprintf('coref_frank_vnew_res_%f_%f_lc%f_se%f', ...
                            lambda_main_coref, lambda_main_face, ...
                            lambda_coref, slack_eps2) '_face1_iter%02d.mat'];

z_strict = train_data_info.z_strict;
max_iter = 5;
for iter = 1:max_iter

  toy_result_file_iter = sprintf(toy_result_file, iter);
  toy_Z_file_iter      = sprintf(toy_Z_file, iter);  

  %% Coref optimization
  train_data_info = updateFaceCorefAddTerms(train_data_info.Y_init,...
                                            train_data_info.Q_init,...
                                            train_data_info.Z_init,...
                                            train_data_info,...
                                            coref_data);
  try
    load(toy_Z_file_iter);
    train_data_info.Z_init = Z_whole;
    fprintf('Loaded iter %d Z file\n', iter);
  catch
    try
      load(toy_result_file_iter);
      fprintf('Loaded iter %d coref data\n', iter);
    catch
      fprintf('Computing iter %d Z file\n', iter);

      add_term = train_data_info.coref_add_term;
      
      % add gender constraints
      add_term = add_term + coref_data.gender_constraints;

      % formualte the coref optimization problem
      [prob_QP, animate_mentions, zinds, ainds, pair_ids] = makeQP_coref(coref_data.pair_features, ...
                                                              coref_data.pair_ids, ...
                                                              coref_data.gcast, ...
                                                              train_data_info.A_ment, ...
                                                              add_term, ...
                                                              lambda_main_coref, lambda_coref, ...
                                                               slack_eps2);

      % sovle the QP for the coref problem
      res = solveQP_coref(prob_QP);
      save(toy_result_file_iter, 'res', 'zinds', 'animate_mentions', 'ainds', 'pair_ids');
    end

    zvals = res(zinds);
    num_cast = size(coref_data.gcast,2);
    num_mentions = numel(animate_mentions);

    Z     = reshape(zvals, [num_cast, num_mentions]);
    Z_whole = zeros(size(coref_data.gcast'));
    Z_whole(:, animate_mentions) = Z;
    znonval = sum(Z_whole,1);

    Z_whole(num_cast, znonval <= 0.999) = 1;
    % proper noun values
    for i = 1:numel(z_strict)
      if (z_strict(i) > 0)
        Z_whole(:,i) = 0;
        Z_whole(z_strict(i), i) = 1;
      end
    end


    %%keyboard;
    save(toy_Z_file_iter, 'Z_whole', 'res');
    train_data_info.Z_init = Z_whole;
  end

  %% Face optimization
  train_data_info = updateFaceCorefAddTerms(train_data_info.Y_init,...
                                            train_data_info.Q_init,...
                                            train_data_info.Z_init,...
                                            train_data_info,...
                                            coref_data);
  toy_Y_file_iter      = sprintf(toy_Y_file, iter);  

  try
    load(toy_Y_file_iter);
    train_data_info.Y_init = Y_whole;
    fprintf('Loaded face file for iter %d\n', iter);
  catch
    size_Y   = size(train_data_info.Y_init);
    prob = makeQP_face(train_data_info, lambda_face_1, lambda_main_face);
    [~, res] = mosekopt('minimize', prob);
    Y_whole  = res.sol.itr.xx(1:(size_Y(1)*size_Y(2)));
    Y_whole  = (reshape(Y_whole, [size_Y(2) size_Y(1)]))';
    pr = evaluateFace(episode_name, train_data_info, Y_whole);
    save(toy_Y_file_iter, 'Y_whole', 'res', 'pr');
    train_data_info.Y_init = Y_whole;
  end 
  
  %% Mapping optimization
  train_data_info = updateFaceCorefAddTerms(train_data_info.Y_init,...
                                            train_data_info.Q_init,...
                                            train_data_info.Z_init,...
                                            train_data_info,...
                                            coref_data);

  train_data_info.Q_init = solveLP_map(train_data_info, coref_data);

end
