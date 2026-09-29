function result = runFFT(x, Fs, nfft)
%RUNFFT Perform and analyze the Fast Fourier Transform.
%
%   result = runFFT(x, Fs, nfft)
%
%   SPANDHAN DSP ANALYSIS
%
%   Performs a one-dimensional FFT and returns both the complete
%   two-sided spectrum and the physically useful one-sided spectrum.
%
%   Mathematical definition:
%
%       X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
%
%   Inputs
%   ------
%   x       : Input signal vector (1-D, real or complex)
%   Fs      : Sampling frequency in Hz
%   nfft    : FFT length.  Use [] for automatic (next power of 2).
%
%   Output
%   ------
%   result is a struct with fields:
%
%     Time-domain
%       .input              Original x (column vector, unchanged)
%       .analysisSignal     DC-removed x used for the FFT
%       .time               Sample time axis (s)
%       .dcValue            Mean value subtracted before analysis
%
%     FFT configuration
%       .nfft               FFT length used
%       .samplingFrequency  Fs (Hz)
%
%     One-sided spectrum   (0 … Fs/2, physically meaningful)
%       .frequency          Frequency axis (Hz)
%       .spectrum           Complex one-sided spectrum X[k]
%       .magnitude          Amplitude spectrum (correctly scaled)
%       .magnitudeDB        20*log10(magnitude)
%       .phase              Unwrapped phase (rad)
%       .power              Power spectrum (magnitude²)
%
%     Two-sided spectrum   (0 … Fs, diagnostic)
%       .frequencyTwoSided
%       .spectrumTwoSided
%       .magnitudeTwoSided
%       .phaseTwoSided
%       .powerTwoSided
%
%     Spectral characteristics
%       .dcMagnitude        |X[0]|/N  — DC component amplitude
%       .peakFrequency      Frequency of dominant non-DC peak (Hz)
%       .peakMagnitude      Amplitude at dominant peak
%       .spectralCentroid   Amplitude-weighted mean frequency (Hz)
%       .spectralBandwidth  Amplitude-weighted std of frequency (Hz)
%       .bandwidth3dB       Width of the -3 dB region (Hz)
%       .lower3dBFrequency  Lower -3 dB edge (Hz)
%       .upper3dBFrequency  Upper -3 dB edge (Hz)
%       .spectralFlatness   Geometric/arithmetic mean ratio ∈ [0,1]
%                           ≈1 → flat (noise-like); ≈0 → tonal
%
%     Signal characteristics
%       .energy             sum(|x_analysis|²)
%       .rms                sqrt(mean(|x_analysis|²))
%
%   Example
%   -------
%       Fs = 16000;
%       t  = (0:1/Fs:2-1/Fs)';
%       x  = sin(2*pi*1000*t) + 0.5*sin(2*pi*3000*t);
%       result = runFFT(x, Fs, 4096);
%
%   One-sided magnitude scaling
%   ---------------------------
%   The raw FFT output is divided by N.  Positive-frequency bins
%   (excluding DC and Nyquist) are then doubled so that the one-sided
%   magnitude equals the true sinusoidal amplitude.
%   For a pure sine of amplitude A: result.peakMagnitude ≈ A.
%
%   DC removal rationale
%   --------------------
%   The mean is removed before the FFT so that a large DC offset does
%   not mask the oscillatory content.  The original signal is stored
%   unchanged in result.input.

    %% ================================================================
    %  1. INPUT VALIDATION
    %  ================================================================

    narginchk(3, 3);

    x = x(:);                           % Force column vector

    if isempty(x)
        error("SPANDHAN:FFT:EmptySignal", ...
            "Input signal cannot be empty.");
    end

    if any(~isfinite(x))
        error("SPANDHAN:FFT:InvalidSignal", ...
            "Input signal contains NaN or Inf values.");
    end

    validateattributes(Fs, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, mfilename, 'Fs');

    %% ================================================================
    %  2. DC REMOVAL
    %  ================================================================
    %
    %  Preserve the caller's signal; analyse a zero-mean copy.

    xOriginal = x;
    dcValue   = mean(x);
    xAnalysis = x - dcValue;

    %% ================================================================
    %  3. DETERMINE FFT LENGTH
    %  ================================================================

    N = length(xAnalysis);

    if isempty(nfft)
        nfftUsed = 2^nextpow2(N);       % Automatic: next power of 2
    else
        validateattributes(nfft, {'numeric'}, ...
            {'scalar', 'integer', 'positive', 'finite'}, mfilename, 'nfft');
        nfftUsed = nfft;
    end

    %% ================================================================
    %  4. TIME AXIS
    %  ================================================================

    time = (0 : N - 1)' / Fs;

    %% ================================================================
    %  5. FFT
    %  ================================================================

    X = fft(xAnalysis, nfftUsed);

    %% ================================================================
    %  6. TWO-SIDED SPECTRUM
    %  ================================================================

    magnitudeTwoSided = abs(X) / N;
    phaseTwoSided     = angle(X);
    powerTwoSided     = abs(X).^2 / N;
    frequencyTwoSided = (0 : nfftUsed - 1)' * Fs / nfftUsed;

    %% ================================================================
    %  7. ONE-SIDED SPECTRUM
    %  ================================================================
    %
    %  For a real signal, all information is contained in bins 0 … Fs/2.
    %  We double the positive-frequency components (except DC and Nyquist)
    %  so that the amplitude corresponds to the true sinusoidal amplitude.

    if rem(nfftUsed, 2) == 0
        % Even length: bins 0, 1 … N/2  (includes Nyquist at N/2)
        positiveBins = 1 : (nfftUsed / 2 + 1);
    else
        % Odd length: bins 0, 1 … (N-1)/2  (no exact Nyquist)
        positiveBins = 1 : ((nfftUsed + 1) / 2);
    end

    XOneSided = X(positiveBins);
    magnitude = abs(XOneSided) / N;

    % Double interior bins to conserve total signal power
    nPos = length(magnitude);
    if nPos > 2
        magnitude(2 : end - 1) = 2 * magnitude(2 : end - 1);
    elseif nPos == 2
        magnitude(2) = 2 * magnitude(2);
    end

    frequency = (0 : nPos - 1)' * Fs / nfftUsed;

    %% ================================================================
    %  8. POWER SPECTRUM
    %  ================================================================

    power = magnitude.^2;

    %% ================================================================
    %  9. MAGNITUDE IN DECIBELS
    %  ================================================================

    magnitudeDB = 20 * log10(max(magnitude, eps));

    %% ================================================================
    %  10. ONE-SIDED PHASE
    %  ================================================================

    phase = unwrap(angle(XOneSided));

    %% ================================================================
    %  11. DC COMPONENT
    %  ================================================================

    dcMagnitude = abs(X(1)) / N;

    %% ================================================================
    %  12. DOMINANT (PEAK) FREQUENCY
    %  ================================================================
    %
    %  Ignore DC (bin 1) so we identify the dominant oscillatory tone.

    if nPos > 1
        searchMag     = magnitude;
        searchMag(1)  = 0;                  % suppress DC
        [peakMagnitude, peakIndex] = max(searchMag);
        peakFrequency = frequency(peakIndex);
    else
        peakMagnitude = magnitude(1);
        peakFrequency = 0;
    end

    %% ================================================================
    %  13. SPECTRAL CENTROID
    %  ================================================================
    %
    %  Amplitude-weighted mean frequency:
    %      f_c = sum(f * |X|) / sum(|X|)

    totalMagnitude = sum(magnitude);

    if totalMagnitude > 0
        spectralCentroid = sum(frequency .* magnitude) / totalMagnitude;
    else
        spectralCentroid = 0;
    end

    %% ================================================================
    %  14. SPECTRAL BANDWIDTH
    %  ================================================================
    %
    %  Amplitude-weighted standard deviation of frequency:
    %      BW = sqrt( sum((f - f_c)^2 * |X|) / sum(|X|) )

    if totalMagnitude > 0
        spectralBandwidth = sqrt( ...
            sum(((frequency - spectralCentroid).^2) .* magnitude) / totalMagnitude);
    else
        spectralBandwidth = 0;
    end

    %% ================================================================
    %  15. -3 dB BANDWIDTH
    %  ================================================================

    peakDB      = max(magnitudeDB);
    indices3dB  = find(magnitudeDB >= peakDB - 3);

    if isempty(indices3dB)
        bandwidth3dB       = NaN;
        lower3dBFrequency  = NaN;
        upper3dBFrequency  = NaN;
    else
        lower3dBFrequency  = frequency(indices3dB(1));
        upper3dBFrequency  = frequency(indices3dB(end));
        bandwidth3dB       = upper3dBFrequency - lower3dBFrequency;
    end

    %% ================================================================
    %  16. SPECTRAL FLATNESS
    %  ================================================================
    %
    %  Ratio of geometric mean to arithmetic mean of the magnitude
    %  spectrum (excluding DC).  Range [0, 1]:
    %      ≈ 1  → spectrally flat (white-noise like)
    %      ≈ 0  → tonal / single dominant frequency

    if nPos > 1
        posMag = max(magnitude(2:end), eps);
    else
        posMag = max(magnitude, eps);
    end

    geometricMean  = exp(mean(log(posMag)));
    arithmeticMean = mean(posMag);

    if arithmeticMean > 0
        spectralFlatness = geometricMean / arithmeticMean;
    else
        spectralFlatness = 0;
    end

    %% ================================================================
    %  17. SIGNAL ENERGY AND RMS
    %  ================================================================

    signalEnergy = sum(abs(xAnalysis).^2);
    signalRMS    = sqrt(mean(abs(xAnalysis).^2));

    %% ================================================================
    %  18. ASSEMBLE RESULT STRUCTURE
    %  ================================================================

    result = struct();

    % --- Time-domain data ---
    result.input          = xOriginal;
    result.analysisSignal = xAnalysis;
    result.time           = time;
    result.dcValue        = dcValue;

    % --- FFT configuration ---
    result.nfft              = nfftUsed;
    result.samplingFrequency = Fs;

    % --- One-sided spectrum ---
    result.frequency   = frequency;
    result.spectrum    = XOneSided;
    result.magnitude   = magnitude;
    result.magnitudeDB = magnitudeDB;
    result.phase       = phase;
    result.power       = power;

    % --- Two-sided spectrum ---
    result.frequencyTwoSided = frequencyTwoSided;
    result.spectrumTwoSided  = X;
    result.magnitudeTwoSided = magnitudeTwoSided;
    result.phaseTwoSided     = phaseTwoSided;
    result.powerTwoSided     = powerTwoSided;

    % --- Spectral characteristics ---
    result.dcMagnitude       = dcMagnitude;
    result.peakFrequency     = peakFrequency;
    result.peakMagnitude     = peakMagnitude;
    result.spectralCentroid  = spectralCentroid;
    result.spectralBandwidth = spectralBandwidth;
    result.bandwidth3dB      = bandwidth3dB;
    result.lower3dBFrequency = lower3dBFrequency;
    result.upper3dBFrequency = upper3dBFrequency;
    result.spectralFlatness  = spectralFlatness;

    % --- Signal characteristics ---
    result.energy = signalEnergy;
    result.rms    = signalRMS;

end
