
% FIlE INITIALISATION 
%PARAMETERS OF THREE TANKS BENCHMARK
global Kp Ki h1c C A1 A2 h1c_0 A1 A2 Qpmax Qf1 h1max Cvb h2max h2c_0 Qf2 h2c_min h2c_max Cvo Cvb
global rate_noise_my1 rate_noise_my2 rate_noise_mUp rate_noise_mQp rate_noise_mUb rate_noise_mUo Te

%PID CONTROLLER PARAMETERS 
Kp=1e-3;%Gain 
Ki=5*1e-6;%Time integration constant 
h1c_0=0; %initiale value of the level in tank T1
h1c=0.5;%set-point of tank T1

%POMPE P1
Qpmax=1e-2; %The maximum flow from pump 1 [m^3/s]

%TANK T1
A1=0.0154; %Cross section of the tank T1 [m^2]
Qf1=1e-4;%Flow leak from tank T1 [m^3/s]
h1max=0.6;%Maximum height of the tank T1 [m]

%VALVE Vb between two tanks
Cvb=1.5938*1e-4;%Global hydraulic flow coefficient of valve Vb
h2max=0.6;%Maximal height of the tank T2 [m]

%TANK T2
A2=0.0154;%Cross section of the tank T2 [m^2]
h2c_0=0.03;%Initial vlue of the level in tank 2
Qf2=1e-4;%Flow leak from tank T2 [m^3/s]

% PARAMETERS OF "ON-OF CONTROLLER"
h2c_min=0.09;%Minimal value of h2 level
h2c_max=0.11;%Maximal value of h2 level

%VALVE Vo (TANK 2 OUTPUT)
Cvo=1.59640*1e-4;%Hydraulic flow coefficient of valve Vo

%OUTPUT PRESSURE TO USER FROM TANK T2
yo=0;

%SENSORS NOISES 
rate_noise_my1=1e-6; %0.5*0.001;
rate_noise_my2=1e-6; %0.3*0.001;
rate_noise_mUp=1e-6; %0.01*1e-5;
rate_noise_mQp=1e-6; %0.01*1e-5;
rate_noise_mUb=0.0000; %0.0000;
rate_noise_mUo=0;

% SAMPLE TIME
Te=1;

