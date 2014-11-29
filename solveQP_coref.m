%%
% brief: Faster iterative optimization to solve QP
% input:
%   - prob_QP: the QP problem in mosek form
%   - max_iter: maximum number of iterations (typically 200 is good for our problem)
% ouptut:
%   - init_val: the solved value of the variable in same format as res.sol.itr.xx from Mosek

function init_val = solveQP_coref(prob_QP, max_iter)

if ~isfield(prob_QP, 'c')
  prob_QP.c = zeros(size(prob_QP.a,2),1);
end

if nargin<2
  max_iter = 200;
end

prob_LP.blc = prob_QP.blc;
prob_LP.buc = prob_QP.buc;
prob_LP.bux = prob_QP.bux;
prob_LP.blx = prob_QP.blx;
prob_LP.a   = prob_QP.a;

Qsparse = sparse(prob_QP.qosubi, prob_QP.qosubj, prob_QP.qoval, numel(prob_QP.c), ...
          numel(prob_QP.c));
Qsparse = (Qsparse + Qsparse' - diag(diag(Qsparse)))*0.5;

[~, res_init] = mosekopt('minimize', prob_LP);

init_val = res_init.sol.itr.xx;
%keyboard;
ubval = init_val'*Qsparse*init_val + prob_QP.c'*init_val;

for i = 1:max_iter
  
  gamma = 2.0/double(i+2);
  
  prob_LP_old = prob_LP;
  prob_LP.c = 2*Qsparse*init_val + prob_QP.c;
  
  [~, res_init] = mosekopt('minimize', prob_LP);

  dual_gap = ubval - res_init.sol.itr.pobjval;  
  %keyboard;
  init_val = init_val + gamma*(res_init.sol.itr.xx - init_val);
  ubval = init_val'*Qsparse*init_val + prob_QP.c'*init_val;
  
  fprintf('iter %d/%d\n', i, max_iter);
  %keyboard;

  %figure(1);
  %hold on;
  %scatter(i, dual_gap, 'g');
  %title(sprintf('iter = %d, dual_gap = %f', i, dual_gap));
  %hold off;

  %figure(2);
  %hold on;
  %scatter(i, ubval, 'r');
  %title(sprintf('iter = %d, primal = %f', i, ubval));
  %hold off;

end
