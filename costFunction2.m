function [jval,grandient]=costFunction2(theta)

jval=(theta(1)-5)^2+(theta(2)-5)^2;
grandient=zeros(2,1);
grandient(1)=2*(theta(1)-5);
grandient(2)=2*(theta(2)-5);