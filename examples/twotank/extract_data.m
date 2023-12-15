% For this script to run one needs to open the simulation window
% and select a type of fault. Then run it once using the launch option
% of the simulation window. Then change the fault_type paramater below
% accordingly.

% parameters
pathB = '../../DS_8_var+residuals/';
fault_type = 'T1_sensor_fault_not_0/';

% for loop
for iter = 1:100
    % run the simulation with random initialisation
    fault_time=randi([0 200]);
    seed=randi(10000000);
    % variable fault intensity (only for T1, T2 leaks & T1 sensor not 0)
    Qf1=5e-5*randn + 1e-4;
    Qf2=5e-5*randn + 1e-4;
    SensorT1Default = 0.1*randn + 0.7;
    sim('CHEM_FULL_BENCHMARK_lmi.slx');
    
    % save the data in the correct folder
    path = strcat(pathB, fault_type, num2str(iter), '/');
    disp([num2str(iter) ' : ' num2str(fault_time) ' ; ' path])
    if ~exist(path, 'dir')
        mkdir(path)
    end
    tmQp = transpose(mQp);
    tmUb = transpose(mUb);
    tmUp = transpose(mUp);
    tmP1 = transpose(mP1);
    tmP2 = transpose(mP2);
    tmy1 = transpose(my1);
    tmy2 = transpose(my2);
    tmQo = transpose(mQo);
    tR1 = transpose(R1);
    tR2 = transpose(R2);
    tR3 = transpose(R3);
    tR4 = transpose(R4);
    tR5 = transpose(R5);
    tR6 = transpose(R6);
    save(strcat(path, 'mQp'), 'tmQp');
    save(strcat(path, 'mUb'),'tmUb');
    save(strcat(path, 'mUp'),'tmUp');
    save(strcat(path, 'my1'),'tmy1');
    save(strcat(path, 'my2'),'tmy2');
    save(strcat(path, 'mP1'),'tmP1');
    save(strcat(path, 'mP2'),'tmP2');
    save(strcat(path, 'mQo'),'tmQo');
    save(strcat(path, 'R1'),'tR1');
    save(strcat(path, 'R2'),'tR2');
    save(strcat(path, 'R3'),'tR3');
    save(strcat(path, 'R4'),'tR4');
    save(strcat(path, 'R5'),'tR5');
    save(strcat(path, 'R6'),'tR6');
    file = fopen(strcat(path, 'fault_time.txt'), 'wt');
    fprintf(file, '%i\n', fault_time);
    fclose(file);
    file = fopen(strcat(path, 'Qf1.txt'), 'wt');
    fprintf(file, '%i\n', Qf1);
    fclose(file);
    file = fopen(strcat(path, 'Qf2.txt'), 'wt');
    fprintf(file, '%i\n', Qf2);
    fclose(file);
end
