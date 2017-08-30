function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%CALCULATE REGCOTFUNCTION (input X=12x2, Y=12x1, theta=2x1)
% Hypothesis

hypothesis=X*theta;% 12x2 *2x1=12x1  % X o day da dc add them 1 cho bias roi
error= (hypothesis-y).^2; % 12x1
J= 1/(2*m)*sum(error); %1x1
%Regulization
theta1=theta;
theta1(1,:)=0; %2x1
theta1=theta1.^2;
regulization=lambda/(2*m)*sum(theta1);
% regulization Cost
J=J+regulization;

%CALCULATE GRADIENT
temp=(hypothesis-y); %12x1
temp= X'*temp; %(12x2)' * 12x1= 2x1 % bao gom buoc nhan va tong sigma luon
%grad=(1/m)*X'*temp;%2x1
grad=(1/m)*temp;
%Calculate regulization
 theta(1,:)=0;
grad_regulization= (lambda/m)*theta; %2x1 % Luu y la bien theta nay dam bao la chua bi thay doi gi
%grad_regulization(1)=0;
%final gradient
grad=grad+grad_regulization;


% =========================================================================

grad = grad(:);

end
