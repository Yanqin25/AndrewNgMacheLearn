function j=costFunction(x,y,theta)

m=size(x,1);
predictions=x*theta;
sqrErrors=(predictions-y).^2;

j=1/(2*m)*sum(sqrErrors);