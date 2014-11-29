function [pr_d_all_old, pr_d_all_new] = checkFace(face_param, coref_param, data_dir)

if nargin < 3
  data_dir = '/scail/scratch/u/vigneshr/joint_nlp_vision/datasets/';
end

iter = 5;

pr_d_all_old = [];
pr_d_all_new = [];
gList = {'highlander_5x14', 'highlander_5x20', 'castle_1x09', ...
      'the_mentalist_1x19', 'californication_1x01'};

for i = 1:numel(gList)
  try
    episode_name = gList{i};
    episode_dir = [data_dir '/' episode_name '/'];
    tt   = load([episode_dir '/data_release/bidirectional_data.mat']);

    toy_Y_file  = [episode_dir sprintf('/bidirectional_results/faces_Y_%f_%f_lf1%f_face1_iter%02d.mat', ...
                   coref_param, face_param, 0.0001, iter)];

    y_new = load(toy_Y_file);    
    pr_old  = evaluateFace(episode_name, tt.train_data_info, tt.train_data_info.Y_init, data_dir);
    pr_new  = evaluateFace(episode_name, tt.train_data_info, y_new.Y_whole, data_dir);
    pr_d_all_old = [pr_d_all_old; pr_old];
    pr_d_all_new = [pr_d_all_new; pr_new];
  catch
    fprintf('%s failed\n', episode_name);
  end
end
