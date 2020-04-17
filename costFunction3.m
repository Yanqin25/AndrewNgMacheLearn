function j=costFunction3(X,y,theta)
m=size(X,1);
predict=X*theta;
sqrErrors=(y-predict).^2;
j=1/(2*m)*sum(sqrErrors);