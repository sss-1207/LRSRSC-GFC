function [Z,E,A,J,W,obj]=LRSRSC_GFC(X, D, lambda, alpha, beta, WW, M)
% This routine solves the following Gaussian Distribution Optimization for Rank Function Approximation problem 
% by using inexact Augmented Lagrange Multiplier, which has been also presented 
% in the paper entitled "Low-rank Sparse Representation Subspace Clustering Algorithm
% with Gaussian Fuzzy Constraint".
%------------------------------
% min ¦µ(¦Òi(Z))+alpha*|M.*Z|_1+lambda*|E|_2,1+ beta*|ZGZ^T|_F^2
% s.t., X = X*Z + E,Z>=0
%         ||
%         ||
%         ||
%¦µ(¦Òi(Z)) = \sum_{i=1^n exp(?¦Òi(Z)) }
%--------------------------------
% min min ¦µ(¦Òi(A))+alpha*|M.*J|_1+lambda*|E|_2,1+ beta*|WGW^T|_F^2
% s.t. X = X*Z + E
%      Z = A
%      Z = J
%      Z = W
%________________________________
% inputs:
%        X -- D*N data matrix, D is the data dimension, M is a fuzzy structured weight matrix, 
% and WW is the data reconstructed from the optimized singular values after the SVD of X.
%% Initializing optimization variables
% intialize the matrices
Wn=size(WW, 1); 
G=(eye(Wn)-WW)*(eye(Wn)-WW)';
nn=size(G, 2);
tol = 1e-6;
max_iter =1e3;
rho = 1.1;
mu =0.1;
max_mu = 1e10;
DEBUG = 1;
loss = 'l21';
[m, n]=size(X);
Z=zeros(n, n);
%A=eye(n);
A=zeros(n, n);
J=zeros(n, n);
W=zeros(n, n);
E=sparse(m, n);

Y1=zeros(m, n);
Y2=zeros(n, n);
Y3=zeros(n, n);
Y4=zeros(n, n);
obj=zeros(1,n);

for iter = 1 : max_iter
Zk = Z;
Ek = E;
Ak = A;
Jk = J;
Wk = W;

%% update A
temp =Z+Y2/mu;
% [~, sigma, ~]=svd(H);
% w=exp(sigma)^2;
w=exp(Z);
w=lambda*(eye(n)-w);
[U, Sigma, V]=svd(temp);
ssigma=Sigma-w/mu;
ssigma=max(0, ssigma);
A=U*ssigma*V';
%% update J
temp = Z+Y3/mu;
J=max(abs(temp)-alpha/mu*M, 0).*sign(temp);
J = max(0, J);
%% update W
W=(Z+Y4/mu)*inv(2*beta/mu*G+eye(nn));
 %% update E
if strcmp(loss, 'l1')
E = prox_l1(A-B*X+Y1/mu, alpha/mu);
elseif strcmp(loss, 'l21')
E = prox_l21(X-D*Z+Y1/mu, 1/mu);
elseif strcmp(loss, 'l2')
E = mu*(X-X*Z+Y1/mu)/(lambda+mu);
else
error('not supported loss function');
end
%% update Z
Z=inv(D'*D+3*eye(n))*(D'*(X-E+Y1/mu)+A+J+W-(Y2+Y3+Y4)/mu);
%% stop criteria   
dY1 = X-D*Z-E;
dY2 = Z-A;
dY3=Z-J;
dY4=Z-W;
chgE = max(max(abs(Ek-E)));
chgZ = max(max(abs(Zk-Z)));
chgA = max(max(abs(Ak-A)));
chgJ = max(max(abs(Jk-J)));
chgW = max(max(abs(Wk-W)));
chg = max([chgE chgZ chgJ chgA chgW max(abs(dY1(:))) max(abs(dY2(:))) max(abs(dY3(:))) max(abs(dY4(:)))]);
if DEBUG
%     if iter == 1 || mod(iter, 10) == 0
        err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2 +norm(dY3,'fro')^2+norm(dY4,'fro')^2);
       % disp('iter=' num2str(iter), 'mu='num2str(mu), 'obj='num2str(obj), 'err='num2str(err));
        obj(iter)=err;
%     end
end
if chg < tol
break;
end
Y1 = Y1 + mu*dY1;
Y2 = Y2 + mu*dY2;
Y3 = Y3 + mu*dY3;
Y4 = Y4 + mu*dY4;
mu = min(rho*mu, max_mu);
end

function x = prox_l1(b,lambda)

% The proximal operator of the l1 norm
% 
% min_x lambda*||x||_1+0.5*||x-b||_2^2

x = max(0,b-lambda)+min(0,b+lambda);

function X = prox_l21(B,lambda)

% The proximal operator of the l21 norm of a matrix
% l21 norm is the sum of the l2 norm of all columns of a matrix 
%
% min_X lambda*||X||_{2,1}+0.5*||X-B||_2^2

X = zeros(size(B));
for i = 1 : size(X,2)
    nxi = norm(B(:,i));
    if nxi > lambda  
        X(:,i) = (1-lambda/nxi)*B(:,i);
    end
end



