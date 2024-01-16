% Steps
% Is the process randomized? -> at least noise should be randomized. Can I
% Is the random seed set somewhere?
% Which parameters in the init can I change to generate different traces?

% Run the simulink model from here

%add constant values to data set
Monit = [Monit, A1*ones(2000,1), Cvb*ones(2000,1)];

%add square root of difference to data set
Monit = [Monit, sqrt(abs(Monit(:,7)))];

% Print data to csv or similar
%header = {'mQp','mUb','Uo','my1','my','h1','mUp','mQ0','y1','vol1','y2','vol2','dy','A1','Cvb','sumsqrt'}; 
header = {'mQp','Uo','my1','h1','mUp','mQ0','y1','vol1','dy','A1','Cvb','ysqrt'};
%h1 seems to be reference temperature

writecell([header; num2cell(Monit)],'data_onetank_withsubst.csv')

% Figure out, which columns could / should be used for model learning