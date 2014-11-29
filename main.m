%%
% brief: runs the bidirectional optimization on the provided dataset
% input:
%   -data_dir: this is the path to the dataset

function main(data_dir)


gList = {'highlander_5x14', 'highlander_5x20', 'castle_1x09', ...
      'the_mentalist_1x19', 'californication_1x01'};

% Ideally you can parallelize this and run the code 
% separatelly for each episode.

for i = 1:5
  runCorefFaceOpt(gList{i}, 10, 0.05, data_dir);
end

[ap_face_unidir, ap_face_bidir] = checkFace(10, 0.05, data_dir);

for i = 1:5
  fprintf('AP for %s = unidir:%f, bidir:%f\n', gList{i}, ap_face_unidir(i).ap, ap_face_bidir(i).ap);
end
