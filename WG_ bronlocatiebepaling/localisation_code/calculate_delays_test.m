function delays = calculate_delays_test(filters,sub);
  
% Examine different ways to calculate delays from filters (maximum value, cross-correlation)
% [e,filters] = compute_delay1_mex(y,speech,L,mu);
  
if nargin < 3,
  listsize = 1;
  if nargin < 2,
    sub = 1;
  end
end
  
L = size(filters,1)/2;
signallength = size(filters,2);
delays = zeros(signallength,3);

previndex = 0;

fprintf('\n');
for i=2*sub:sub:signallength,
  filt1 = filters(1:L,i);
  filt2 = filters(L+1:2*L,i);
  
  % Biased correlation
  corr = xcorr(filt1,filt2);  
  [m,previndex] = max(corr);
  
  xx = [max(1,previndex-1):min(previndex+1,length(corr))]';
  yy = corr(xx);
  par = [xx.*xx xx ones(size(xx))]\yy;
  if par(1) == 0,
    delays(i,1) = previndex-L;
  else
    delays(i,1) = -par(2)/(2*par(1))-L;
  end
  
  % Unbiased correlation
  %corr = xcorr(filt1,filt2,'unbiased');
  %[m,index] = max(corr);
  %delays(i,2) = index-L;
  
  % Maximum (quadratic interpolation)
  [m,index1] = max(filt1);     
  [m,index2] = max(filt2);
  
  if (index1 ~= 1) & (index1 ~= L),
    xx = [index1-1:index1+1]';
    par = [xx.*xx xx ones(3,1)]\filt1(xx);
    if par(1) ~= 0,
      index1 = -par(2)/(2*par(1));
    end
  end
  
  if (index2 ~= 1) & (index2 ~= L),
    xx = [index2-1:index2+1]';
    par = [xx.*xx xx ones(3,1)]\filt2(xx);
    if par(1) ~= 0,
      index2 = -par(2)/(2*par(1));
    end
  end
  
  delays(i,3) = index1-index2;
  
  screen(floor(100*i/signallength));
end

  