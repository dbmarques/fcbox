% FCSPECTRUM() - Computes the frequency spectrum of a diversity of 
%   synchrony estimates between two time series, representing their 
%   "functional connectivity". It can compute: magnitude-squared coherence,
%   amplitude correlations, power correlations, phase locking value 
%   (PLV; aka mean phase coherence [MPC]), imaginary coherence, phase lag 
%   index (PLI), weighted phase lag index (wPLI), and debiased weighted 
%   phase lag index (dwPLI). 
%   It can compute based on different methods of signal segmenting:
%   Welch's method, Chronux multitaper method, trials as segments, and
%   permutation of trials as segments.
% 
%   Usage:
%       [FC,MPD,f] = fcspectrum(x,y,window,noverlap,nfft,Fs,fcest,freqrange,bandlim,surr);
%
%   Inputs:
%       x           = signal 1 (samples,trials)
%       y           = signal 2 (samples,trials)
%       window      = window vector. This input determines the segmentation
%                       method. If window is a vector of the same length as
%                       the signals, it considers the trials ('trials') as 
%                       windowed segments. If window is a vector of 
%                       different size, it performs Welch's method 
%                       ('welch') of windowed segment averaging. If window 
%                       has two values, it performs multitaper 
%                       ('dpss') method using discrete prolate spheroidal 
%                       (Slepian) sequences as performed in the Chronux 
%                       toolbox, using the first value as time 
%                       half-bandwidth and the second as number of tapers. 
%                       [default: hann(length(x))]
%       noverlap    = ratio of window length (<1) or number of samples of 
%                       overlap between segments for Welch's method. If not
%                       this method, this input is ignored.
%                       [default: 0.5]
%       nfft        = number of points for FFT 
%                       [default: 2^nextpow2(length(window))]
%       Fs          = sampling rate (Hz)         
%                       [default: 1]
%       fcest       = string definig the estimate to compute: 
%                       'mscohere': magnitude-squared coherence. As
%                                   computed by the MATLAB built-in
%                                   mscohere function
%                       'ampcorr': amplitude correlations
%                       'powcorr': power correlations
%                       'plv': phase locking value/mean phase coherence. As
%                                   described by Lachaux et al. (1999).
%                                   This estimate is called "mpc" in
%                                   Marques et al. (2022 JNeurosci).
%                       'icoh': imaginary coherence
%                       'pli': phase lag index 
%                       'wpli': weighted phase lag index (Vinck's estimates 
%                                   but used another reference)
%                       'dwpli': debiased weighted phase lag index. As
%                                   described by Vinck et al. (2011). This
%                                   computation is based on the FieldTrip 
%                                   toolbox.
%                       [default: 'mscohere']
%       freqrange   = vector defining the limits of the frequency range 
%                       [fmin fmax]. If empty, returns the FC across the 
%                       whole spectrum (0 to Nyquist)       
%       bandlim     = vector or array defining the bands' limits
%                       (bands,[fmin fmax]). If empty, returns the FC
%                       across the whole spectrum. If inputed, returns the
%                       bands' averages. the band limits must be within the
%                       range specified by freqrange.
%       surr        = integer or boolean deciding to calculate surrogate FC
%                       as a "null distribution" instead of real FC. 
%                       Obtains FC of different trials across iterations.
%                       To compute both and statistical comparisons, use 
%                       fcspectrum2. 
%                       surr == 0: does not calculate null FC.
%                       surr == 1: calculates FC for every trial to a 
%                           different trial. 
%                       surr > 1: calculates FC across 'surr' number of
%                           randomly selected two different
%                           trials (e.g. 200).
%                       [default: 0]
% 
%   Outputs:
%       FC          = "functional connectivity" spectra (freqs,trials)
%       MPD         = mean phase difference spectra (freqs,trials)
%       f           = frequency vector (Hz)
% 
%   To develop:
%       Compute specific frequencies
%       Time-frequency computations over overlapped moving windows
%       Same computations for instantaneous bandpass-filtered signals
% 
% Author: Danilo Benette Marques, 2018
% Last update: 2025-04-02

function [FC,MPD,f] = fcspectrum(x,y,window,noverlap,nfft,Fs,fcest,freqrange,bandlim,surr)

if size(x) ~= size(y)
    error('x and y must have the same sizes')
end

% % Set parameters
if nargin<3 | isempty(window)
    L = size(x,1);
    window = hann(L);
end

if numel(window) == 2
    segmethod = 'dpss';
    TW = window(1);
    K = window(2);
elseif length(window)==size(x,1)
    segmethod = 'trials'; %{'trials','permtrials'}
else
    segmethod = 'welch';
end

if nargin<4 | isempty(noverlap)
    noverlap = .5;
end

if nargin<5
    nfft = [];
end

if nargin<6 | isempty(Fs)
    Fs = 1;
end

if nargin<7 | isempty(fcest)
    fcest = 'mscohere';
end

if nargin<8 | isempty(freqrange)
    freqrange = [0 Fs/2]; %freq. range: 0 to Nyquist
end

if nargin<9 | isempty(bandlim)
    isband = false;
else
    isband = true;
end

if nargin<10 | isempty(surr)
    surr = false;
end

% % Set adjusted parameters
%Sizes
N = size(x,1);
Ntrials = size(x,2);

%Get normalized slepian sequences for multitapers
if strcmp(segmethod,'dpss')
    window = dpss(N,TW,K); 
    window=window*sqrt(Fs); %as computed in Chronux
end

%window length
L = length(window);

%Number of segments
switch segmethod
    case 'welch'
        Nwin = (N-L)/(L-noverlap)+1;
    case 'dpss'
        Nwin = size(window,2);
    case 'trials'
        Nwin = Ntrials;
        Ntrials = 1;
    case 'permtrials' 
        warninig('permtrials option is under developtment. not recommended')
        Nwin = Ntrials;
        Ntrials = 200; %number of iterations
end

%if noverlap is ratio
if noverlap<1
    noverlap = round(noverlap*L);
end

if isempty(nfft) 
    nfft = 2^nextpow2(L);
    % OBS: the easiest way to fft specified freqs is to run the default 
    % with  specified nfft and then run interp1(f,xfft,foi,'nearest') 
    % across frequencies-of-interest. Then just needs to get the 'fidx' of 
    % the same size of foi
    % 
    % Example:
    % dfoi = 0.2;
    % foi = 0:dfoi:Fs/2;
    % xfft = interp1(f,xfft,foi,'nearest');
end

%Get frequency vector
df = Fs/nfft;
f = 0:df:Fs; %symmetrical spectrum
f = f(1:nfft); %if uneven

fidx = f>=freqrange(1) & f<=freqrange(2);
f = f(fidx); %for even: f(1:nfft/2+1), or odd: f(1:(nfft+1)/2)

%Set indices and number of iterations for surrogate FC
if surr==1 %one random different trial to every single trial
    notshuffled = 1;
    while notshuffled
        idx_trials_surr(:,1) = 1:Ntrials;
        idx_trials_surr(:,2) = datasample(1:Ntrials,Ntrials,'replace',false); %random for every trial
        notshuffled=any(idx_trials_surr(:,1)==idx_trials_surr(:,2)); %assure different trials
    end

elseif surr>1 %specified number of iterations
    Niter = surr; 

    notshuffled = 1;
    while notshuffled
        idx_trials_surr(:,1) = datasample(1:Ntrials,Niter,'replace',true);
        idx_trials_surr(:,2) = datasample(1:Ntrials,Niter,'replace',true); %two random trials
        notshuffled=any(idx_trials_surr(:,1)==idx_trials_surr(:,2)); %assure different trials per row
    end

    %Define surr number of trials/windows
    if any(strcmp(segmethod,{'welch','dpss'}))
        Ntrials = Niter;
    elseif strcmp(segmethod,'trials')
        Nwin = Niter;
    end
end

% % Run computations of FC
for trial = 1:Ntrials

    %Set trials indices
    if surr==false
        trialx = trial;
        trialy = trial;
    %Shuffled x and y trials to get "null" FC
    elseif surr>=true  
        if any(strcmp(segmethod,{'welch','dpss'}))
            trialx = idx_trials_surr(trial,1);
            trialy = idx_trials_surr(trial,2);
        end    
    end

    %Get windowed segments
    clear xwin ywin
    switch segmethod
        case 'welch'
            %Welch's method
            for win = 1:Nwin
                start = (win-1)*(L-noverlap)+1;
                stop = start+L-1;
        
                xwin(:,win) = window.*x(start:stop,trialx);
                ywin(:,win) = window.*y(start:stop,trialy);
            end

        case 'dpss'
            %Chronux multitaper method
            for win = 1:Nwin
                xwin(:,win) = window(:,win).*x(:,trialx);
                ywin(:,win) = window(:,win).*y(:,trialy);
            end

        case 'trials'
            %Trials are considered windowed segments
            for win = 1:Nwin

                if surr==false %trials(windowed seg) indices
                    trialx = win;
                    trialy = win;
                elseif surr==true %shuffled trials
                    trialx = idx_trials_surr(win,1);
                    trialy = idx_trials_surr(win,2);
                end

                xwin(:,win) = window.*x(:,trialx);
                ywin(:,win) = window.*y(:,trialy);
            end

        case 'permtrials' 
            %Permutation of trials considered windowed segments
            permtrials = datasample(1:Nwin,Nwin); %random indices
            for win = 1:Nwin
                xwin(:,win) = window.*x(:,permtrials(win));
                ywin(:,win) = window.*y(:,permtrials(win));
            end

    end

    %Get segments spectra
    xwin_fft = fft(xwin,nfft,1);
    ywin_fft = fft(ywin,nfft,1);
    xywin_fft = xwin_fft.*conj(ywin_fft);

    %Compute mean phase difference
    mpd = mean( angle(xywin_fft) ,2);
    mpd = mpd(fidx,:);

    %Compute specific FC estimate
    switch fcest

        case 'mscohere' 
            % Magnitude squared coherence
            Pxx = mean(xwin_fft.*conj(xwin_fft),2);    
            Pyy = mean(ywin_fft.*conj(ywin_fft),2);   
            Pxy = mean(xwin_fft.*conj(ywin_fft),2);
    
            coh = (abs(Pxy).^2)./(Pxx.*Pyy);

        case 'ampcorr'
            % Amplitude correlation
            magx = abs(xwin_fft);    
            magy = abs(ywin_fft);    

            C = corr(magx',magy'); %correlation matrix
            ampcorr = C(logical(eye(size(C,1)))); %identity: corr(x(f[i]),y(f[i]))

            coh = ampcorr;

        case 'powcorr'
            % Power correlation
            pxx = xwin_fft.*conj(xwin_fft);    
            pyy = ywin_fft.*conj(ywin_fft);    

            C = corr(pxx',pyy'); %correlation matrix
            powcorr = C(logical(eye(size(C,1)))); %identity: corr(x(f[i]),y(f[i]))

            coh = powcorr;
        
        case 'plv' 
            % Mean phase coherence/Phase Locking Value
            plv = abs( mean( exp(1i*angle(xywin_fft)) ,2) );  

            coh = plv;    

        case 'icoh'
            % Imaginary coherence
            Pxx = mean(xwin_fft.*conj(xwin_fft),2);    
            Pyy = mean(ywin_fft.*conj(ywin_fft),2);   
            Pxy = mean(xwin_fft.*conj(ywin_fft),2);
            
            icoh = (imag(Pxy).^2)./(Pxx.*Pyy);

            coh = icoh;

        case 'pli'
            % Phase-lag index
            pli  = abs(mean(sign(imag(xywin_fft)),2));

            coh = pli;

        case 'wpli'
            % Weighted phase-lag index 
            wpli = abs( sum( abs(imag(xywin_fft)).*sign(imag(xywin_fft)) ,2)...
                ./sum( abs(imag(xywin_fft)) ,2) );

            % % as computed in FieldTrip
            %imagsum      = nansum(imag(xywin_fft),2);
            %imagsumW     = nansum(abs(imag(xywin_fft)),2);
            %wpli  = imagsum./imagsumW; %obs: has negative values, but abs is equal to math formula

            coh = wpli;

        case 'dwpli'
            % Debiased weighted phase-lag index 
            % OBS: as computed in FieldTrip
            imagsum      = nansum(imag(xywin_fft),2);
            imagsumW     = nansum(abs(imag(xywin_fft)),2);
            debiasfactor = nansum(imag(xywin_fft).^2,2);
            dwpli  = (imagsum.^2 - debiasfactor)./(imagsumW.^2 - debiasfactor);

            coh = dwpli;

    end
    coh = coh(fidx,:); 

    %Band average
    if isband
        for band = 1:size(bandlim,1)
            Coh(band,:) = nanmean(coh(f>=bandlim(band,1) & f<=bandlim(band,2),:),1);
            Mpd(band,:) = nanmean(mpd(f>=bandlim(band,1) & f<=bandlim(band,2),:),1);
        end
        coh = Coh;
        mpd = Mpd;
    end

    %Concatenate final output
    FC(:,trial) = coh;
    MPD(:,trial) = mpd;
end

end