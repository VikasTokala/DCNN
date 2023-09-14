function X = welchspsd(x, fs, nfft, varargin)

if nargin<4
    plt = 0;
else
    plt = varargin{1};
end

if nargin<5
    lpfil = 0;
else
    lpfil = varargin{2};
end

% [x,fs] = v_readwav(xpath, 'g');

frame_len = nfft;
frame_inc = round(0.2*frame_len);

w = sqrt(hanning(frame_len,'periodic'));
win_anal{1} = w ./ sqrt(sum(w(1:frame_inc:frame_len).^2 * frame_len * frame_inc));
win_anal{2} = w .* sqrt(frame_len * frame_inc / sum(w(1:frame_inc:frame_len).^2)); %

[X_tmp,X_ref_tail_anal,pmX] = stft_v2('fwd',x,win_anal, frame_inc, frame_len, fs);

X = mean(X_tmp,3);


if plt==1
    
    freqs = 0:fs./nfft:(fs/2);
    
    if lpfil==0
            figure;
            plot(freqs, 10*log10(X.*conj(X)));
            title("Welch's periodogram")
    else
            fullpsd = X.*conj(X);
            lppsd = filter(1./lpfil*ones(1,lpfil), 1, fullpsd);
            figure;
            plot(freqs, 10*log10(lppsd))
            title("LPF Welch's periodogram");
    end
    xlabel('Frequency (Hz)');
end