function [ J,grad ] = costFunctiontest( theta, X, y )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%Tinh ham cost cua LOGISTIC THEO EUCLIDEAN DISTANCE DE QUAN SAT CAI SAI
%tinh ham sigmoid - ham predicitons
predictions= X*theta;% predictions(mx1) of hypothesis on all m examples, X=100*3
sqrErrors = (predictions-y).^2; %squared errors
J=1/(2*m)*sum(sqrErrors);

%compute gradient descent 
% cost ban dau dung repmat va .*
%     Errors = (predictions-y); % errors (mx1)  
%     Errors=repmat(Errors,1,size(X,2)); % mo rong ma tran a bang n features
%     %Errrors m*n (m=100sample n=3features)
%     sigma= Errors.*X; %mxn (100x3)
%     sigma=(sum(sigma))';%(1x3)->(3x1)
%     grad= (1/m)*sigma;  %3x1

    Beta = (predictions-y); % errors (mx1)    
    grad= (1/m)*X'*Beta;  %3x1
 

% =============================================================

end

