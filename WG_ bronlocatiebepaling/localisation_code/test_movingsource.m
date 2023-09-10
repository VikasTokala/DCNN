%%%%%%%%%%%%%%%%%%%%%
% Signal generation %
%%%%%%%%%%%%%%%%%%%%%

% Parameters
fs = 8000;          % Sampling frequency [Hz]
totallength = 30;   % Signal length [s]
snr0 = 40;          % SNR-value [dB]
signallength = fs*totallength;

sub = 200;          % Subsampling for calculating/plotting delays
bound = 1;          % Negative/Positive boundary for anomaly detection [samples]

% Room + microphone positions
T60 = 250;                              % Reverberation time [ms]
Rdim = [4 7 2.75];                      % Room dimensions [m]
Pmic = [1.7 3.5 1.375;2 3.5 1.375];     % Microphone positions [m]
nrmics = size(Pmic,1);

% Moving speech source
file_speech = 'd:/sounds/vlopwosp/track01_cont.wav';
R = 1.5;    % Radius
f = 0.0529; % Time-variation factor
t = [0:0.1:totallength];
Psrc = [2+R*cos(2*pi*f*t') 3.5+R*sin(2*pi*f*t') 1+t'/30];   % Position of speech source [m]

% Correct delays
cdelays = correct_delay(Pmic,Psrc,fs);
cdelays = frame2signal(cdelays,1,signallength/length(t)*1000);cdelays=cdelays(1:signallength);

% Generate speech signal
s0 = wavread(file_speech);
s0 = resample(s0,fs,44100);
s0 = repmat(s0,ceil(signallength/size(s0,1)),1);
s0 = s0(1:signallength);

volume = Rdim(1)*Rdim(2)*Rdim(3);
area = 2*(Rdim(1)*Rdim(2)+Rdim(1)*Rdim(3)+Rdim(2)*Rdim(3));
Reflec = exp(-0.163*volume/(area*T60/1000)); % Eyring's formula
N = 1000;

for i=1:nrmics,
  [s(:,i),H] = simnonstat(s0,Pmic(i,:),Psrc,Rdim,Reflec*ones(6,1),N,t,fs);
end

speech = ones(signallength,1);  % Continuous adaptation

% Noise signal (white noise)
n = randn(signallength,nrmics);

% Noisy signals
fac = norm(s(:,1))/(norm(n(:,1))*sqrt(10^(snr0/10)));
n = n*fac;
y = s+n;

%%%%%%%%%%%%%%%%%%
% ALGORITHM: GCC %
%%%%%%%%%%%%%%%%%%

N = 256;            % Framesize
lambda = 0.95;      % Averaging factor
weighting = 1;      % PHAT-weighting
wintype = 1;        % Hanning window
intfactor = 5;      % Interpolation factor

delays = gcc(y,N,lambda,weighting,wintype,N/2,2*N,intfactor);

delays = frame2signal(delays,1,signallength/length(delays)*1000);
delays = delays(1:signallength);

plot(cdelays(sub:sub:end));hold on;plot(-delays(sub:sub:end),'g.');hold off
axis([0 signallength/sub -10 10]);

[rmse,anom] = delay_perf(-delays(sub:sub:end),cdelays(sub:sub:end),[-bound bound])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ALGORITHM: AEDA (original) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L = 32;         % Filterlength
mu = 0.01;      % Stepsize
option = 1;     % Normalisation per channel
iter = 1;       % Number of iterations
algo = 2;

[e,filters] = compute_delay1_mex(y,speech,L,mu,option,iter,1,algo);
delays = calculate_delays_test(filters,sub);

plot(cdelays(sub:sub:end));hold on;plot(-delays(sub:sub:end,1),'g.');hold off
axis([0 signallength/sub -10 10]);

[rmse,anom] = delay_perf(-delays(sub:sub:end,1)',cdelays(sub:sub:end),[-bound bound])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ALGORITHM: AEDA (+reinitialisation) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L = 32;         % Filterlength
mu = 0.01;      % Stepsize
option = 1;     % Normalisation per channel
iter = 1;       % Number of iterations
algo = 2;
len = 2000;     % Length for adaptation

filters = zeros(2*L,floor((signallength-len+1)/sub));
for t=len:sub:signallength,
  [e,tmp] = compute_delay1_mex(y(t-len+1:t,:),speech(t-len+1:t),L,mu,option,iter,1,algo);
  filters(:,t/sub) = tmp(:,end);
end 
delays = calculate_delays_test(filters,1);

plot(cdelays(sub:sub:end));hold on;plot(-delays(:,1),'g.');hold off
axis([0 signallength/sub -10 10]);

[rmse,anom] = delay_perf(-delays(:,1)',cdelays(sub:sub:end),[-bound bound])
      
