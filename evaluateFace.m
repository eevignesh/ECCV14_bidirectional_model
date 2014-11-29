function pr = evaluateFace(episode_name, train_data_info, Y, data_dir)

if nargin < 4
  data_dir = '/scail/scratch/u/vigneshr/joint_nlp_vision/datasets/';
end

full_episode_dir = [data_dir episode_name];
annotation_file =  [full_episode_dir '/data_release/face_annotations.mat'];

[max_val, max_ids] = max(Y);
[~, sortids]       = sort(max_val, 'descend');

try
  load(annotation_file);
catch
  fprintf('No face annotation available for this episode\n');
  pr = [];
  return;
end

annoids = annoids(train_data_info.all_fids);

corr_vals = 0;
numtot    = 0;
for i = 1:numel(sortids)
  if(max_ids(sortids(i))==annoids(sortids(i)))
    corr_vals = corr_vals + 1;
  end

  numtot      = numtot + 1;

  num_corr(numtot) = corr_vals;
  prec(numtot)     = corr_vals/numtot;
  rec(numtot)      = numtot;
end
fprintf('Num total annotated faces = %f\n', numtot);

rec = rec/numtot;

acc = corr_vals/numtot;
ap  = mean(prec);
pr.acc  = acc;
pr.ap   = ap;
pr.prec = prec;
pr.rec  = rec;

