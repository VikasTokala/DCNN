% function y=frame2signal(x,Fs,Tframe)
%
% Transforms frame-based parameters to signal-based parameters
% i.e. repeat every value N times, where N is the framelength
% 
% AUTHOR 	Simon Doclo, 11/05/98
%
% OUTPUT 	y	: signalvector
%			 
% INPUTS 	x 	: framevector
%		Fs	: sampling frequency (Hz)
%		Tframe	: framelength (ms)
%
% USES
%
% USED BY	spchdet_off
%

function y=frame2signal(x,Fs,Tframe);

framelength=ceil(Fs*Tframe/1000);
y(1:framelength*length(x))=0;

for i=1:length(x),
  y((i-1)*framelength+1:i*framelength)=x(i);
end