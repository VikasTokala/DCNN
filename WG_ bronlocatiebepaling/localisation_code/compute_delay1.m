% [e,filters] = compute_delay1(y,speech,L,mu,option,iter,sub,algo)
% 
% Computes time-delays between 2 different microphone signals 
% Technique: adaptive eigenvalue decomposition (J. Benesty)
% 
% AUTHOR	Simon Doclo, 11/01/01
% OUTPUT	e             : error signal
%               filters       : different estimated impulse responses
%
% INPUTS        y             : microphone signals
%               speech        : speech/noise detection per sample (only adaptation during speech)
%               L             : lengths of the different filters
%               mu            : stepsize parameter for LMS-procedure (can be larger than 2!)
%               option        : normalisation (0: no normalisation - default, 1: normalisation per channel, 2: global normalisation)
%               iter          : number of iterations (optional, default 1)
%               sub           : subsampling factor (optional, default 1)
%               algo          : 1 = LMS-based subspace tracking procedure (smallest eigenvalue is zero) (default)
%                               2 = LMS-based subspace tracking procedure

function [e,filters] = compute_delay1(y,speech,L,mu,option,iter,sub,algo);

signallength = size(y,1);
ep = 1e-10;

if nargin < 8,
  algo = 1;
  if nargin < 7,
    sub = 1;
    if nargin < 6,
      iter = 1;
      if nargin < 5,
        option = 0;
      end
    end
  end
end

if isempty(speech),
  speech = ones(signallength,1);
end

% Algorithm
% Initialisation 
u = [zeros(ceil(L/2)-1,1);1;zeros(floor(L/2),1); zeros(L,1)]; % Eigenvector (unit norm)
e = zeros(floor(signallength/sub),1);
filters = zeros(2*L,floor(signallength/sub));
  
% No adaptation during first L-1 samples
x1 = zeros(L,1); % Data vectors 
x2 = zeros(L,1);
for i=1:L-1,   
  x1 = [y(i,1);x1(1:L-1)]; % Update data vector per channel
  x2 = [y(i,2);x2(1:L-1)]; % Update data vector per channel
end
 
fprintf('\n');

for i=L:signallength, 
  
  x1 = [y(i,1);x1(1:L-1)]; % Update data vector per channel
  x2 = [y(i,2);x2(1:L-1)]; % Update data vector per channel
  
  if (rem(i,sub) == 0) & (speech(i) == 1), % Subsampling, adaptation during speech periods
      
    teller = i/sub;
    xtot = [-x2;x1]; 
    
    % Normalisation
    if option == 1,
      xtot = [xtot(1:L)/(norm(xtot(1:L))+ep);xtot(L+1:2*L)/(norm(xtot(L+1:2*L))+ep)];
    elseif option == 2,
      xtot = xtot/(norm(xtot)+ep);
    end
    
    % Adaptation (LMS-like, approximation if eigenvalue is zero)
    for k=1:iter,
      e(teller) = u'*xtot;
      if algo == 1,
        u = u - (mu*e(teller))*xtot;
      elseif algo == 2,
        u = u - (mu*e(teller))*(xtot-e(teller)*u);
      end
      u = u/norm(u);
    end

    % Calculate delays
    filters(:,teller) = u;
    
  end % subsample + speech
         
  if (rem(i,1000) == 0),
      screen(floor(100*i/signallength));
  end
    
end % signallength
  
