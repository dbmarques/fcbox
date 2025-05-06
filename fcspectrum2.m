% FCSPECTRUM2() - Computes functional connectivity spectrum of statistical
%   differences against a null distribution. Uses the function and the 
%   same parameters of 'fcspectrum'.
% 
% Author: Danilo Benette Marques, 2025
% Last update: 2025-03-19

function [tvalue,pvalue,f,FC,FCsurr] = fcspectrum2(x,y,window,noverlap,nfft,Fs,fcest,freqrange,bandlim)

%Leave empty to use default fcspectrum params
if nargin<3
    window = [];
end
if nargin<4
    noverlap = [];
end
if nargin<5
    nfft = [];
end
if nargin<6
    Fs = [];
end
if nargin<7
    fcest = [];
end
if nargin<8
    freqrange = [];
end
if nargin<9
    bandlim = [];
end

%Run real and surrogate FC
[FC,~,f] = fcspectrum(x,y,window,noverlap,nfft,Fs,fcest,freqrange,bandlim);
[FCsurr] = fcspectrum(x,y,window,noverlap,nfft,Fs,fcest,freqrange,bandlim,true);

%T-tests
[~,pvalue,~,stat] = ttest2(FC',FCsurr');
pvalue = pvalue';
tvalue = [stat.tstat]';

% end