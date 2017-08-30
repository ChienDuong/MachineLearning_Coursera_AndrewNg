function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%Part 1 feedforward then compute costfunction tra ve gia tri J

%theta1 = 25x401
%theta2= 10x26
%Add ones to the X data matrix
X = [ones(m, 1) X]; %X = 5000x401
% calculate hiddenlayer
 z2=X*Theta1'; % 5000*401  (25*401)'  = 5000*25
 a2=sigmoid(z2);
 a2=[ones(m, 1) a2]; %5000*26
%calculate output layer
z3= a2*Theta2'; %5000*26 (10*26)' = 5000*10;
a3=sigmoid(z3); % a3 chinh la hypothesis 5000*10
% TINH COST
%Prepare data y, m la so luong sample

%#1 solution but slow. tao ma tran 0, roi cai gi giong vs y(i) thi cho
%bang1
% yv=zeros(m,num_labels); % cho nay quan trong, output dataset phai du voi tat ca samples va labels
% for i=1:m % m la so class
% yv(i,y(i))=1;
% end
%yv=5000*10

%#2 solution, faster
yv=bsxfun(@eq,y,1:num_labels);


%calculation  Costfunction
    Error=-yv.*log(a3)-(1-yv).*log(1-a3);% 5000*10 .* 5000*10=5000*10
    % 5000hang la so sample, 10cot la so class
    Error=Error(:);
    Error=sum(Error);
    J=Error/m;

%REGULARIZED COST FUNCTION
%theta1= 25*401 lop input, theta2=10x26 lop hidden
%Phai giua gia tri Theta1 , Theta2 vi ve sau con xai, ko dc set = 0
ThetaRegulization1=Theta1;
ThetaRegulization2=Theta2;
ThetaRegulization1(:,1)=0;
ThetaRegulization2(:,1)=0;
%regulization tren cac layer
Theta=[ThetaRegulization1(:);ThetaRegulization2(:)]; % giong nn_params nhung ma cua phan bias cho ve 0 roi
reg_layer=Theta.^2;
reg_layer=sum(reg_layer);

% reg_layer1=Theta1.^2;
% reg_layer1=reg_layer1(:);
% reg_layer1=sum(reg_layer1);
% 
% reg_layer2=Theta2.^2;
% reg_layer2=reg_layer2(:);
% reg_layer2=sum(reg_layer2);
% 
% regulization=lambda/(2*m)*(reg_layer1+reg_layer2);

regulization=lambda/(2*m)*reg_layer;
%Output: regularized cost funciton
J=J+regulization;
% -------------------------------------------------------------
%PART2:Backpropagaion
delta_Accumulate2=zeros(num_labels,hidden_layer_size+1);%10x26
delta_Accumulate1=zeros(hidden_layer_size,input_layer_size+1);%25x401

for t=1:m
%Step1: feedforward, nhin vao hinh ex3 forwward propagation
a1=X(t,:);%1x401
z2=a1*Theta1'; % 1*401  (25*401)'  = 1*25
a2=sigmoid(z2);%1x25
a2=[1 a2]; %1*26
%calculate output layer
z3= a2*Theta2'; %1*26 (10*26)' = 1*10;
a3=sigmoid(z3); % a3 chinh la hypothesis 1*10

%Step2: tinh sai so o lop ra, tu day den het PART 2 nhin vao ex4 hinh back
delta3=a3-yv(t,:);% 1x10, hang 1 cua y, sai so tai 10 not cua ngo ra

%Step3: Compute another layer
z2=[1 z2];%1x26, them ca bias vao, neu bo dong nay thi ko can tinh 2:end o dong 147
delta2=(delta3*Theta2).*sigmoidGradient(z2);% delta2=1x10*10x26 =1x26.*1x26, 
delta2=delta2(2:end); % do remove cai tinh cho bias
%Step4: calculate accumulation
%tich luy tac dong len gradient cua tung parameter the ta=> theta 2 =
%num_units lop 3 x num_units lop 2 +1 = 10*26
delta_Accumulate2=delta_Accumulate2+delta3'*a2;% (1x10)'* 1x26=10x26
delta_Accumulate1=delta_Accumulate1+delta2'*a1;%(1x25)'*1x401=25x401% bo theta 0
end

%Step 5: Calculate gradient
Theta1_grad=1/m*delta_Accumulate1;%+lambda/m*(Theta1);%1/m*25x401 + lambda/m* 25x401 voi cot 1 =0, ung voi bias cua layer input 
Theta2_grad=1/m*delta_Accumulate2;%+lambda/m*(Theta2);%10x26
% =========================================================================

%PART3:compute regulization
regulization1=lambda/m*(ThetaRegulization1);
regulization2=lambda/m*(ThetaRegulization2);
%final gradient
Theta1_grad=Theta1_grad+regulization1;
Theta2_grad=Theta2_grad+regulization2;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
