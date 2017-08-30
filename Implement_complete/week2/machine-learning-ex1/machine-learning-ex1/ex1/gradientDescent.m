function [theta, theta_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
 %J_history = zeros(num_iters, 1);
theta_history = zeros(num_iters, 2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %Chi tiet coi o file huong dan
    hypothesis= X*theta; % predictions(mx1) of hypothesis on all m examples
    Errors = (hypothesis-y); % errors (mx1)  
    temp= X'*Errors  %2*1
    theta= theta-(1/m)*alpha*temp;

    % ============================================================

    % Save the cost J in every iteration    
   % J_history(iter) = computeCost(X, y, theta); 
      theta_history(iter,:) = theta'; % tam thoi thay cost = theta de bieu dien figure 3
end

end
