function [cdelays,source_distance,mic_distance,list] = correct_delay(pos_mics,pos_source,fs,option)

% [cdelays,source_distance,mic_distance,list] = correct_delay(pos_mics,pos_source,fs,option)
%
% Computes correct time delays between source and microphone array
% 
% AUTHOR	Simon Doclo, 11/01/01
% OUTPUT	cdelays         : time difference in samples (between microphones in list)
%               source_distance : distance between source and microphones
%               mic_distance    : microphone distance (between microphones in list)
%               list            : list of microphone numbers
% INPUTS        pos_mics        : position of microphone array [x1 y1 z1;x2 y2 z2;...;xN yN zN]
%               pos_source      : position of source [x y z] (Mx3-matrix)
%               fs              : sampling frequency
%               option          : 1 = delays between microphones and first microphone (default)
%                                 2 = delays between all microphones
%

if nargin < 4,
  option = 1;
end

NrChannels = size(pos_mics,1); % number of microphones
M = size(pos_source,1); % number of source positions
c = 340; % speed of sound (m/s)

if option == 1,
  list = [[2:NrChannels]' ones(NrChannels-1,1)];
elseif option == 2,
  list =fliplr(nchoosek([1:NrChannels],2));
end
listsize = size(list,1);

for i=1:NrChannels,
  for m=1:M,
    source_distance(i,m) = norm(pos_source(m,:)-pos_mics(i,:));
  end
end

for k = 1:listsize,
  i = list(k,1);
  j = list(k,2);
  mic_distance(k,1) = norm(pos_mics(i,:)-pos_mics(j,:));
  for m=1:M,
    cdelays(k,m) = (source_distance(i,m)-source_distance(j,m))*fs/c;
  end
end


