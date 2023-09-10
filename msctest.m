% MSC test script

clc;
clearvars;
close all;


[asig,fs] = audioread('/Users/vtokala/Documents/Research/di_nn/Dataset/clean_testset_1f/p278_051_0.wav');

mscohere(asig(:,1),asig(:,2),hamming(512),400,512,fs);