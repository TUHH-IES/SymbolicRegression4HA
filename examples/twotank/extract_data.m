% parameters
pathB = './19-01-2024/';
example = 'single_tank';
seed=randi(10000000);
is_single = true;

file = strcat(pathB, "/", example, '.csv');
if ~exist(pathB, 'dir')
    mkdir(pathB)
end

if is_single
    header = {'mQp','Uo','my1','h1','mUp','mQ0','y1','vol1','dy','A1','Cvb','ysqrt'};
else
    header = {'mQp','mUb','Uo','my1','my','h1','mUp','mQ0','y1','vol1','y2','vol2','dy','A1','Cvb','sumsqrt'}; 
end
writecell(header,file)

% for loop
for iter = 1:100
    % run the simulation with random initialisation
    run("INIT.m")
    h1c_0=rand(1)/0.5;
    h2c_0=rand(1)/0.5;
    sim(strcat(example,'.slx'));
   
    Monit = [Monit, A1*ones(2000,1), Cvb*ones(2000,1)];
    if is_single
        %add square root of difference to data set
        Monit = [Monit, sqrt(abs(Monit(:,7)))];
    else
        Monit = [Monit, sqrt(abs(Monit(:,9)-Monit(:,11)))]; 
    end
    writematrix(Monit,file,'WriteMode','append');
    clearvars -except pathB example seed is_single file
end
