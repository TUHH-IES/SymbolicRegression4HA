% Steps
% Is the process randomized? -> at least noise should be randomized. Can I
% Is the random seed set somewhere?
% Which parameters in the init can I change to generate different traces?

% Run the simulink model from here

% Print data to csv or similar
header = {'mQp','mUb','Uo','my1','my','h1','mUp','mQ0','y1','vol1','y2','vol2'};
%h1 seems to be reference temperature

writecell([header; num2cell(Monit)],'data_nonoise.csv')

% Figure out, which columns could / should be used for model learning