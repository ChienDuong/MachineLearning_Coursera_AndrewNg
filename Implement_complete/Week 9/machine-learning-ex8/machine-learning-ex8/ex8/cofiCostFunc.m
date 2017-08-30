function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features 5x3
%        Theta - num_users  x num_features matrix of user features 4x3
%        Y - num_movies x num_users matrix of user ratings of movies 5x4
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the  5x4
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X 5x3
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%                     4x3
%

%Calculate cost function%Y=5x4
%Note: cai quan trong la co ma tran theta voi X roi cu tinh predictions het
% nhung chi cong nhung gia tri nao ma co R=1, tuc la co user rate

predictions=X*Theta';%5x3*(4x3)'=5x4
temp=(predictions-Y).^2;%5x4 .Tinh toan bo predictions
temp=temp.*R;%5x4 % chi chon ra nhung cai co user vote
J=1/2*sum(sum(temp));% tong hang roi tong cot sum(sum(J))=sum(4x1)=1 so
%Regulization
regulization= lambda/2*sum(sum(Theta.^2))+lambda/2*sum(sum(X.^2));
%Final cost
J=J+regulization;


%Calculate gradient
%x_gradient, fearture gradient 5x3
for i=1:num_movies
   %xpredictions=X(i,:)*(Theta)'%1x3*(4x3)'=1x4

   temp1=(predictions(i,:)-Y(i,:));%1x4':' dai dien cho j user=> movies *users
   temp1=temp1.*R(i,:); % 1x4 tai 1  movies, chi chon cai ma dc user vote
   %theta=user *feartures
   temp1=temp1*Theta; %1x4*4x3=1x3 : 3 feartures theo cot, movies*feartures
   X_grad(i,:)=temp1;%5x3
end
%regulization term
x_regulization=lambda*X;

X_grad=X_grad+x_regulization;
%theta gradient, user gradient 4x3
for i=1:num_users
   temp2=(predictions(:,i)-Y(:,i));%5x1 ':' dai dien cho j user=> movies *users
   temp2=temp2.*R(:,i); % 5x1 tai 1  movies, chi chon cai ma dc user vote
   %theta=user *feartures
   temp2=(temp2)'*X; %(5x1)'*5x3=1x3 : 3 feartures theo cot, movies*feartures
   Theta_grad(i,:)=temp2;
    
end
%regulization theta
theta_regulization=lambda*Theta;
%Final gradient
Theta_grad=Theta_grad+theta_regulization;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
