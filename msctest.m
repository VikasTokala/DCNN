% MSC test script

clc;
clearvars;
close all;


[asig,fs] = audioread('/Users/vtokala/Documents/Research/di_nn/Dataset/clean_trainset_1f/p226_014_2.wav');
nfft=512;
mscohere(asig(:,1),asig(:,2),hanning(512),400,512,fs);
% lapsd = welchspsd(asig(:,1),fs,512,1,0);
% rapsd = welchspsd(asig(:,2),fs,512,1,0);
% 
% cpsd = lapsd .* conj(rapsd);
% 
% msc = abs(cpsd).^2 ./ ((lapsd .* rapsd) + 1e-8);
% 
% freqs = 0:fs/nfft:(fs/2);
% msc=mean(msc,2);
% figure;
% plot(freqs, msc);
% title("Welch's p
% eriodogram based MSC")
figure()
cpsd(asig(:,1),asig(:,1),hanning(512),400,512,'',fs);
figure()
cpsd(asig(:,2),asig(:,2),hanning(512),400,512,'',fs);
figure()
cpsd(asig(:,1),asig(:,2),hanning(512),400,512,'',fs);