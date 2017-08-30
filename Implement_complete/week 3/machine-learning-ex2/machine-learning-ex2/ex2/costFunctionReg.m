function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


%tinh ham sigmoid - ham predicitons
    z= X*theta;% predictions(mx1) of hypothesis on all m examples 
    %118x28 * 28*1=118*1
    predictions= sigmoid(z);  %(mx1) =118*1
% tinh regulization nhu theo de bai
    theta1=theta;
    theta1(1)=0;
    regulization=(lambda/(2*m))*sum(theta1.^2);
%tinh cai ben trong tong sigma
    logErrors=y.*log(predictions)+(1-y).*log(1-predictions);
%tra ve gia tri cost function
    J=-1/m*sum(logErrors)+regulization;

%compute gradient descent 
    Errors = (predictions-y); % errors (mx1)  
    Errors=repmat(Errors,1,size(X,2)); % mo rong ma tran a bang n features
    %Errrors m*n (m=118 sample n=28features) => X=118*28
    sigma= Errors.*X; %mxn (118x28)
    sigma=(sum(sigma))';%(1x28)->(28x1)
    grad= (1/m)*sigma;  %28x1
    %Tinh regulization 
    gra_reguliztion=(lambda/m)*theta;
    gra_reguliztion(1)=0; % 28*1 voi phan tu dau =0
    %gradient final
    grad=grad+gra_reguliztion; % hai cai 28*1 voi nhau

% =============================================================

end
