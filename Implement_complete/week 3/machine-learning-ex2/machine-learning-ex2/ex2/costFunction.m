function [J, grad] = costFunction(theta, X, y)
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
%
%tinh ham sigmoid - ham predicitons
z= X*theta;% predictions(mx1) of hypothesis on all m examples, X=100*3
predictions= sigmoid(z);  %(mx1) =100*1
%ko can tinh regulization nhu theo de bai
%tinh cai ben trong tong sigma
sqrErrors=y.*log(predictions)+(1-y).*log(1-predictions);
%tra ve gia tri cost function
J=-1/m*sum(sqrErrors);

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
