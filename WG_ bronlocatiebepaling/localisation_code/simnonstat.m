% [y,H]=simnonstat(x,Pmic,Psrc,Rdim,Reflec,N,t,fs);
%
% non-stationary acoustics simulation (moving omni-directional source)
%
% INPUTS
%	x	: input signal
%	Pmic	: vector containing the 3-dim coordinates of the microphone
%	Psrc    : Mx3-matrix containing the coordinates of the source as a function of 
%		  time. Each time instance the source moves, a new coordinate set is added
%	Rdim    : room dimensions (see simroom)
%	Reflect : reflection coefficients (see simroom)
%	N       : acoustic transfer function length
%	t	: time vector (s) of length M, corresponding to Psrc, t(1)=0;
%	fs      : sampling frequency (Hz)
% OUTPUT
%	y	: microphone signal
%       H       : room impulse responses

function [y,H]=simnonstat(x,Pmic,Psrc,Rdim,Reflec,N,t,fs);

H=zeros(N,length(t));
for i=1:length(t);
	H(:,i)=simroommex(Pmic,Psrc(i,:),Rdim,Reflec,N,fs);
end
y=zeros(length(x)+N-1,1);
inc=[1:length(t)];
fprintf('\n');
for n=1:length(x)
	i=max(inc.*(n/fs>t));
	y(n:n+N-1)=y(n:n+N-1)+x(n)*H(:,i);
	if rem(n,1000) == 0,
	  screen(floor(100*n/length(x)));
	end
	
end
y=y(1:length(x));
