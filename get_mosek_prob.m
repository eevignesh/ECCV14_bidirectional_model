%%
% code adapted from Piotr Bojanowski

function prob = get_mosek_prob(AA, BB, Bs, lambda, face_add_term, isQP)


% variable being solved for 
% Y, eps, tau (p*num_bags)
% [Y(1,:) Y(2,:) ... Y(p, :), eps(1, :) .... eps(p, :), tau(:)]
% minimize ||[eps; kappa*tau]||^2_F - lambda*Trace(Y*face_add_term)
% or minimize ||eps||^2_F - kappa*1^T*(tau)
% Y^T.AA - BB >= tau
% Bs*Y^T = eps
% 0 <= Y <= 1
% tau <= 0

% BB = Z*CC

%n = numel(IA{1});
%p = numel(IB{1});

kappa = 1e8;

n = size(AA,1);
p = size(BB,1);
num_bags = size(BB,2);

num_pos = size(face_add_term,1);
assert(size(face_add_term,2)==p);
face_add_term = [face_add_term; zeros((n-num_pos), p)];


%AA = cell2mat(IA');
%BB = cell2mat(IB');
Bs = double(Bs);
[ij, ik, val] = find(Bs);
asubi = zeros(numel(ik)*p, 1);
asubj = asubi;
for i = 1:p
  asubi(((i-1)*numel(ik)+1):(i*numel(ik))) = ij + n*(i-1);
  asubj(((i-1)*numel(ik)+1):(i*numel(ik))) = ik + n*(i-1);
end
aval = repmat(val, [p 1]);
%keyboard;
A_eq2 = [sparse(asubi, asubj, aval), -sparse(1:n*p, 1:n*p, 1), sparse(n*p, p*num_bags)];

%keyboard;
%Q = kron(eye(p), Bs);
%keyboard;
%% QP formulaiton
if (isQP)
  prob.qosubi = n*p + (1:(n*p));
  prob.qosubj = prob.qosubi;
  prob.qoval = ones(1, numel(prob.qosubi));
  prob.c = [-face_add_term(:)*lambda; sparse(n*p,1); -sparse(kappa*ones(num_bags*p, 1))];
  prob.bux = [sparse(ones(n*p,1)); inf(n*p,1); sparse(num_bags*p, 1)];
else
%% norm formulaiton
  prob.qosubi = n*p + (1:(n*p + num_bags*p));
  prob.qosubj = prob.qosubi;
  prob.qoval = [ones(1, n*p) kappa*ones(1, num_bags*p)];
  prob.c = [-face_add_term(:)*lambda; sparse(n*p,1); -sparse(num_bags*p, 1)];
  %prob.c = sparse(2*n*p + num_bags*p, 1);
  prob.bux = [sparse(ones(n*p,1)); inf(n*p,1); inf(num_bags*p, 1)];
end

%[prob.qosubi, prob.qosubj, prob.qoval] = find(tril(kron(eye(p), Bs)));
%keyboard;


%A_eq3 = kron([zeros(1, 2) 1 zeros(1, 17)], ones(1,n));
%A_eq3 = [A_eq3; kron([zeros(1, 8) 1 zeros(1, 11)], ones(1,n))];
%b_leq3 = [0;0]; b_req3 = [0;0];

A_eq  = kron(ones(1, p), eye(n));
b_leq = ones(n,1);
b_req = ones(n,1);

%keyboard;
%A_eq  = [A_eq; A_eq3];
%b_leq = [b_leq; b_leq3];
%b_req = [b_req; b_req3];

A = kron(eye(p), AA');
BB = BB';
blc = BB(:);
buc = inf(size(blc));

A1 = [A_eq sparse(size(A_eq,1), n*p) sparse(size(A_eq,1), p*num_bags); 
      A sparse(size(A,1), n*p) -sparse(1:(num_bags*p), 1:(num_bags*p), 1)];
%keyboard;
prob.a = [A1;A_eq2];

prob.blc = [sparse([b_leq; blc]); sparse(n*p,1)];
prob.buc = [sparse([b_req; buc]); sparse(n*p,1)];
%prob.blc = [sparse(b_leq); sparse(n*p,1)];
%prob.buc = [sparse(b_req); sparse(n*p,1)];

prob.blx = [sparse(n*p,1); -inf(n*p,1); -inf(num_bags*p,1)];
%keyboard;

