function result = runSTFT(x, Fs, windowLength, overlap, nfft)
%RUNSTFT Perform Short-Time Fourier Transform analysis.
%
%   result = runSTFT(x, Fs, windowLength, overlap, nfft)
%
%   SPANDHAN DSP ANALYSIS
%
%   Divides the input signal into overlapping windowed frames and
%   computes an FFT for each frame, producing a time-frequency map
%   (spectrogram) of how spectral content evolves over time.
%
%   Mathematical definition:
%
%       STFT{x[n]}(m, ω) = Σ_n  x[n] · w[n − m·R] · exp(−j·ω·n)
%
%   where:
%       w[n]  = Hann analysis window
%       m     = frame index
%       R     = hop size = windowLength − overlap
%       ω     = normalised digital frequency
%
%   Inputs
%   ------
%   x             : Input signal vector (1-D)
%   Fs            : Sampling frequency in Hz
%   windowLength  : Number of samples per STFT window
%   overlap       : Number of overlapping samples between windows
%   nfft          : FFT length per window (must be ≥ windowLength)
%
%   Output
%   ------
%   result is a struct with fields:
%
%     Time-domain
%       .input                 Original x (column vector, unchanged)
%       .analysisSignal        DC-removed x used for the STFT
%       .dcValue               Mean subtracted before analysis
%
%     STFT core data
%       .spectrum              Complex spectrogram S (nfft/2+1 × nFrames)
%       .frequency             Frequency axis F (Hz), length = nfft/2+1
%       .time                  Time axis T (s), one value per frame
%       .magnitude             |S|
%       .magnitudeDB           20·log10(max(|S|, ε))
%       .phase                 unwrap(∠S, [], 2) — unwrapped along time
%
%     Window configuration
%       .window                Hann window vector used
%       .windowLength          Samples per window
%       .overlap               Overlapping samples
%       .hopSize               windowLength − overlap
%       .nfft                  FFT length
%
%     Sampling
%       .samplingFrequency     Fs (Hz)
%
%     Per-frame features
%       .dominantFrequency     Hz of the max-magnitude bin per frame (col vector)
%       .dominantMagnitude     Magnitude at the dominant bin per frame
%       .spectralCentroid      Amplitude-weighted mean frequency per frame (Hz)
%       .spectralBandwidth     Amplitude-weighted std of frequency per frame (Hz)
%       .frameEnergy           sum(|S[:,m]|²) per frame
%
%     Global energy
%       .totalEnergy           sum of all frame energies
%
%     Resolution
%       .timeResolution        windowLength / Fs  (seconds)
%       .frequencyResolution   Fs / nfft          (Hz per bin)
%
%   Example
%   -------
%       Fs = 16000;
%       t  = (0:1/Fs:3-1/Fs)';
%       x  = chirp(t, 500, 3, 5000, "linear");
%       result = runSTFT(x, Fs, 1024, 512, 2048);
%
%   Window choice
%   -------------
%   A periodic Hann window is used for all spectral analysis.  It
%   gives good side-lobe attenuation (~31 dB) with a relatively
%   narrow main lobe, making it the standard choice for STFT.
%   The "periodic" option ensures the FFT shift is correct.
%
%   Time–frequency trade-off
%   ------------------------
%   Longer window  → finer frequency resolution, coarser time resolution
%   Shorter window → finer time resolution, coarser frequency resolution
%   This is the Heisenberg–Gabor uncertainty principle for signals.

    %% ================================================================
    %  1. INPUT VALIDATION
    %  ================================================================

    narginchk(5, 5);

    x = x(:);

    if isempty(x)
        error("SPANDHAN:STFT:EmptySignal", ...
            "Input signal cannot be empty.");
    end

    if any(~isfinite(x))
        error("SPANDHAN:STFT:InvalidSignal", ...
            "Input signal contains NaN or Inf values.");
    end

    validateattributes(Fs, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, mfilename, 'Fs');

    validateattributes(windowLength, {'numeric'}, ...
        {'scalar', 'integer', 'positive', 'finite'}, mfilename, 'windowLength');

    validateattributes(overlap, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'overlap');

    validateattributes(nfft, {'numeric'}, ...
        {'scalar', 'integer', 'positive', 'finite'}, mfilename, 'nfft');

    %% ================================================================
    %  2. PARAMETER CONSISTENCY CHECKS
    %  ================================================================

    if windowLength < 2
        error("SPANDHAN:STFT:InvalidWindowLength", ...
            "Window length must be at least 2 samples.");
    end

    if overlap >= windowLength
        error("SPANDHAN:STFT:InvalidOverlap", ...
            "overlap (%d) must be less than windowLength (%d).", ...
            overlap, windowLength);
    end

    if nfft < windowLength
        error("SPANDHAN:STFT:InvalidFFTLength", ...
            "nfft (%d) must be >= windowLength (%d).", ...
            nfft, windowLength);
    end

    N = length(x);

    if N < windowLength
        error("SPANDHAN:STFT:ShortSignal", ...
            "Signal length (%d samples) is shorter than window length (%d samples).", ...
            N, windowLength);
    end

    %% ================================================================
    %  3. DC REMOVAL
    %  ================================================================
    %
    %  Remove the mean so the DC bin does not dominate the spectrogram.
    %  The original signal is preserved in result.input.

    xOriginal = x;
    dcValue   = mean(x);
    xAnalysis = x - dcValue;

    %% ================================================================
    %  4. WINDOW AND HOP SIZE
    %  ================================================================
    %
    %  Periodic Hann window:
    %    hann(L,"periodic") = sin²(π·n/L),  n = 0…L-1
    %  Minimises spectral leakage while maintaining a well-defined
    %  periodicity assumption for each FFT frame.

    window  = hann(windowLength, "periodic");
    hopSize = windowLength - overlap;

    %% ================================================================
    %  5. COMPUTE STFT VIA SPECTROGRAM
    %  ================================================================
    %
    %  spectrogram() applies the Hann window to each overlapping frame,
    %  zero-pads to nfft, and computes the DFT.
    %  Output S is (nfft/2+1) × nFrames for real input.

    [S, F, T] = spectrogram(xAnalysis, window, overlap, nfft, Fs);

    %% ================================================================
    %  6. MAGNITUDE
    %  ================================================================

    magnitude = abs(S);

    %% ================================================================
    %  7. MAGNITUDE IN DECIBELS
    %  ================================================================

    magnitudeDB = 20 * log10(max(magnitude, eps));

    %% ================================================================
    %  8. PHASE
    %  ================================================================
    %
    %  Unwrap along the time axis (dimension 2) for a smooth phase track.

    phase = unwrap(angle(S), [], 2);

    %% ================================================================
    %  9. DOMINANT FREQUENCY PER FRAME
    %  ================================================================
    %
    %  The bin with the highest magnitude in each frame.
    %  DC suppression: zero bin 1 before the search.

    searchMag         = magnitude;
    searchMag(1, :)   = 0;              % suppress DC bin in all frames

    [dominantMagnitude, dominantIdx] = max(searchMag, [], 1);

    dominantFrequency = F(dominantIdx);

    dominantFrequency = dominantFrequency(:);
    dominantMagnitude = dominantMagnitude(:);

    %% ================================================================
    %  10. SPECTRAL CENTROID PER FRAME
    %  ================================================================
    %
    %  Amplitude-weighted mean frequency:
    %      fc(m) = Σ_k f(k)·|S(k,m)| / Σ_k |S(k,m)|

    frequencyCol           = F(:);
    totalMagnitudePerFrame = sum(magnitude, 1);      % 1 × nFrames

    nFrames        = size(magnitude, 2);
    spectralCentroid = zeros(nFrames, 1);

    validFrames = totalMagnitudePerFrame > 0;

    spectralCentroid(validFrames) = ( ...
        sum(frequencyCol .* magnitude(:, validFrames), 1) ./ ...
        totalMagnitudePerFrame(validFrames) )';

    %% ================================================================
    %  11. SPECTRAL BANDWIDTH PER FRAME
    %  ================================================================
    %
    %  Amplitude-weighted standard deviation of frequency:
    %      BW(m) = sqrt( Σ_k (f(k)-fc(m))²·|S(k,m)| / Σ_k |S(k,m)| )

    spectralBandwidth = zeros(nFrames, 1);

    for m = 1:nFrames
        if totalMagnitudePerFrame(m) > 0
            fc                  = spectralCentroid(m);
            spectralBandwidth(m) = sqrt( ...
                sum(((frequencyCol - fc).^2) .* magnitude(:, m)) / ...
                totalMagnitudePerFrame(m));
        end
    end

    %% ================================================================
    %  12. FRAME ENERGY AND TOTAL ENERGY
    %  ================================================================

    frameEnergy = sum(abs(S).^2, 1)';   % column vector, one value per frame
    totalEnergy = sum(frameEnergy);

    %% ================================================================
    %  13. RESOLUTION METRICS
    %  ================================================================

    timeResolution      = windowLength / Fs;    % seconds
    frequencyResolution = Fs / nfft;            % Hz per bin

    %% ================================================================
    %  14. ASSEMBLE RESULT STRUCTURE
    %  ================================================================

    result = struct();

    % --- Time-domain data ---
    result.input          = xOriginal;
    result.analysisSignal = xAnalysis;
    result.dcValue        = dcValue;

    % --- STFT core data ---
    result.spectrum    = S;
    result.frequency   = F;
    result.time        = T;
    result.magnitude   = magnitude;
    result.magnitudeDB = magnitudeDB;
    result.phase       = phase;

    % --- Window configuration ---
    result.window        = window;
    result.windowLength  = windowLength;
    result.overlap       = overlap;
    result.hopSize       = hopSize;
    result.nfft          = nfft;

    % --- Sampling ---
    result.samplingFrequency = Fs;

    % --- Per-frame features ---
    result.dominantFrequency = dominantFrequency;
    result.dominantMagnitude = dominantMagnitude;
    result.spectralCentroid  = spectralCentroid;
    result.spectralBandwidth = spectralBandwidth;
    result.frameEnergy       = frameEnergy;

    % --- Global energy ---
    result.totalEnergy = totalEnergy;

    % --- Resolution ---
    result.timeResolution      = timeResolution;
    result.frequencyResolution = frequencyResolution;

end
