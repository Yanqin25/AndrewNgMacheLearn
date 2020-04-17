function [jval,gradient]=costFunctionLogic(theta)
X=[1,2,3;1,3,4;2,1,2;4,5,6;3,5,3;1,7,2];
y=[1;1;1;0;0;0];
jval=-y'*log(hypothetical(X,theta))-(1-y)'*log(1-hypothetical(X,theta));
gradient=zeros(3,1);
gradient(1)=1/6*((hypothetical(X,theta) -y)'*X(:,1));
gradient(2)=1/6*((hypothetical(X,theta) -y)'*X(:,2));
gradient(3)=1/6*((hypothetical(X,theta) -y)'*X(:,3));