function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
%dim= size(X,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%J=(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))/m+lambda/(2*m)*theta(2:dim,:)'*theta(2:dim,:);
J=(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))/m+lambda/(2*m)*theta(2:end,:)'*theta(2:end,:);
%grad(1)=(sigmoid(X*theta)-y)'*X(:,1)/m;
%for n =2:dim
  %grad(n)=(sigmoid(X*theta)-y)'*X(:,n)/m+lambda/m*theta(n);
%endfor
grad=((sigmoid(X*theta)-y)'*X./m)'+lambda/m*theta;
grad(1)=(sigmoid(X*theta)-y)'*X(:,1)/m;



% =============================================================

end

