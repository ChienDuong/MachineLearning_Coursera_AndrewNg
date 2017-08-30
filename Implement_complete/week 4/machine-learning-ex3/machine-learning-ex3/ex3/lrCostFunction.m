function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
%tinh ham sigmoid - ham predicitons
z= X*theta;%5*4 * 4*1 predictions(mx1) of hypothesis on all m examples, z=5*1
predictions= sigmoid(z);  %(mx1) =5*1
%tinh regulization nhu theo de bai
    theta1=theta;
    theta1(1)=0;
    regulization=(lambda/(2*m))*sum(theta1.^2);
%tinh cai ben trong tong sigma
Errors=y.*log(predictions)+(1-y).*log(1-predictions);% y=5*1
%tra ve gia tri cost function
J=-(1/m)*sum(Errors)+regulization;

%%%%%%%%%%%%%compute gradient descent%%%%%%%%%%%%%%%%%%%%%% 

    Beta = (predictions-y); % errors (mx1)    
    grad= (1/m)*X'*Beta;  %4x5*5x1=4x1
 %Tinh regulization 
    gra_reguliztion=(lambda/m)*theta; % chinh la bien themp nhu tren goi y
    gra_reguliztion(1)=0; 
 %gradient final
    grad=grad+gra_reguliztion; % hai cai 28*1 voi nhau



% =============================================================

grad = grad(:);

end
