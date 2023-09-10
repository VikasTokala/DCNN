% [estdel,ccw] = gcc(y,N,lambda,weighting,wintype,overlap,L,intfactor);
%
% Compute delay between 2 signals using Generalised Crosscorrelation
% Using zero-padding in frames (circular convolution effects)
%
% Reference: C.H. Knapp and G.C. Carter, "The Generalized Correlation Method for Estimation of Time Delay," 
%            IEEE Trans. Acoust., Speech and Signal Proc., vol. 24, no. 4, Aug. 1976, pp. 320-327.
%
% OUTPUT    estdel         Estimated delay (number of samples)
%           ccw            Weighted time-domain cross-correlation  
%
% INPUTS    y              signal
%           N              framesize
%           lambda         averaging factor for spectra (lambda = 0, no averaging)
%           weighting      Weighting function (optional)
%                            0: No weighting
%                            1: PHAT, Phase Transform (default)
%                            2: ML, Maximum Likelihood
%                            3: SCOT, Smoothed Coherence Transform
%                            4: Roth
%                            5: Eckart
%           wintype     Windowing function for computing psd/coherence (optional)
%                            0: Rectangular window (default)
%                            1: Hanning window
%                            2: Hamming window
%           overlap        number of overlap samples (optional, default N/2)
%           L              size of FFT (optional, default 2^(nextpow2(N)+1))
%           intfactor      Interpolation factor for GCC function (optional, default 1)
%                            
  
function [estdel,ccw,C1,C2,CC,coh,CCw] = gcc(y,N,lambda,weighting,wintype,overlap,L,intfactor);

totallength = size(y,1);  
  
if nargin < 8,
  intfactor = 1;
  if nargin < 7,
    L = 2^(nextpow2(N)+1);
    if nargin < 6,
      overlap = N/2;
      if nargin < 5,
        wintype = 0;
        if nargin < 4,
          weighting = 1;
        end
      end
    end
  end
end

% Windowing
if wintype == 0,
  win = ones(N,1);
elseif wintype == 1,
  win = hanning(N);
elseif wintype == 2,
  win = hamming(N);
end

NrFrames = ceil((totallength-N)/overlap);
framestart = 1;

C1 = zeros(L,NrFrames);
C2 = zeros(L,NrFrames);
CC = zeros(L,NrFrames);
coh = zeros(L,NrFrames);
coh2 = zeros(L,1);
ccw = zeros(L*intfactor,NrFrames);
CCw = zeros(L,NrFrames);
estdel = zeros(NrFrames,1);

% Initialisation using first frame
frame1 = y(framestart:framestart+N-1,1).*win;
frame2 = y(framestart:framestart+N-1,2).*win;
F1 = fft(frame1,L);
F2 = fft(frame2,L);
C1(:,1) = abs(F1.*conj(F1)); % PSD1 (real)
C2(:,1) = abs(F2.*conj(F2)); % PSD2 (real)
CC(:,1) = F1.*conj(F2); % Cross-correlation
framestart = framestart+overlap;

for i=2:NrFrames,
  
  frame1 = y(framestart:framestart+N-1,1).*win;
  frame2 = y(framestart:framestart+N-1,2).*win;

  % Frequency-domain correlation
  F1 = fft(frame1,L);
  F2 = fft(frame2,L);

  C1(:,i) = lambda*C1(:,i-1) + (1-lambda)*abs(F1.*conj(F1)); % PSD1 (real)
  C2(:,i) = lambda*C2(:,i-1) + (1-lambda)*abs(F2.*conj(F2)); % PSD2 (real)
  CC(:,i) = lambda*CC(:,i-1) + (1-lambda)*F1.*conj(F2); % Cross-correlation
  coh(:,i) = CC(:,i)./sqrt(C1(:,i).*C2(:,i)); % Coherence 

  % Weighting
  if weighting == 0,
    weight = ones(L,1);
  elseif weighting == 1,
    weight = 1./abs(CC(:,i));
  elseif weighting == 2,
    coh2 = abs(coh(:,i)).^2; % Magnitude-squared coherence
    weight = coh2./(abs(CC(:,i)).*(1-coh2));
  elseif weighting == 3,
    weight = 1./sqrt(C1(:,i).*C2(:,i));
  elseif weighting == 4,
    weight = 1./C1(:,i);
  elseif weighting == 5,
    weight = abs(CC(:,i))./((C1(:,i)-abs(CC(:,i))).*(C2(:,i)-abs(CC(:,i))));
  end

  CCw(:,i) = CC(:,i).*weight;

  % Estimation of delay

  tmp = fftshift(real(ifft(CCw(:,i))));

  if intfactor == 1;
    ccw(:,i) = tmp;
    [m,index] = max(ccw(:,i));
    estdel(i) = index-L/2-1;
  else
    ccw(:,i) = interp(tmp,intfactor);
    [m,index] = max(ccw(:,i));
    estdel(i) = (index-1)/intfactor-L/2;
  end

  framestart = framestart+overlap;
end
