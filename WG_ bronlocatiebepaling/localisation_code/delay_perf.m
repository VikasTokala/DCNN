% [rmse,anom] = delay_perf(delays,cdelay,bound,sub)
%
% Compute performance of TDE algorithm
%
% OUT    rmse       Mean-square error
%        anom       Percentage wrong detection
% IN     delays     Estimated delays
%        cdelay     Correct delay (either number of vector)
%        bound      Negative/Positive boundary for anomaly detection
%        sub        subsampling factor (optional, default 1)
%

function [rmse,anom] = delay_perf(delays,cdelay,bound,sub);

if nargin < 4,
  sub = 1;
end

error = delays-cdelay; % Works both for scalar as vector cdelay
error = error(sub:sub:end);
signallength = length(error); 

rmse = sqrt((norm(error)^2)/signallength);

count = zeros(signallength,1);
count(find(error < bound(1) | error > bound(2))) = 1;
anom = sum(count)/signallength;

