function result = runWavelet(x, Fs, waveletName, decompositionLevel)
%RUNWAVELET Multi-resolution analysis using the Discrete Wavelet Transform.
%
%   result = runWavelet(x, Fs, waveletName, decompositionLevel)
%
%   SPANDHAN DSP — Multi-Resolution Time-Scale Analysis
%
%   Decomposes a signal into approximation and detail components at
%   multiple dyadic scales.  Unlike FFT (global frequency content) or
%   STFT (windowed frequency content), DWT reveals *what signal
%   structures exist at different time scales*, making it the ideal
%   tool for transient events such as impulses and step transitions.
%
%   Decomposition tree (decompositionLevel = 5):
%
%       Input
%         ├── D1   (highest frequency, finest time resolution)
%         ├── D2
%         ├── D3
%         ├── D4
%         ├── D5
%         └── A5  (lowest frequency, coarsest time resolution)
%
%   Approximate frequency bands at Fs = 16000 Hz, level 5:
%
%       D1  4000–8000 Hz       A5  0–250 Hz
%       D2  2000–4000 Hz       D5  250–500 Hz
%       D3  1000–2000 Hz       D4  500–1000 Hz
%
%   Inputs
%   ------
%   x                  : Input signal (1-D vector)
%   Fs                 : Sampling frequency in Hz
%   waveletName        : Wavelet family string, e.g. "db4" (default),
%                        "sym4", "coif5".  Daubechies-4 is a good
%                        general-purpose choice: compact support, good
%                        transient representation, short filter.
%   decompositionLevel : Number of decomposition levels (default 5).
%                        Automatically clamped to wmaxlev() if too large.
%
%   Output
%   ------
%   result is a struct with fields:
%
%     Time-domain
%       .input                   Original x (column vector, unchanged)
%       .analysisSignal          DC-removed x
%       .time                    Sample time axis (s)
%
%     Configuration
%       .samplingFrequency       Fs
%       .waveletName             String
%       .decompositionLevel      Levels actually used (≤ requested)
%       .maximumLevel            wmaxlev() for this signal and wavelet
%
%     DWT outputs (from wavedec)
%       .coefficients            C  — packed coefficient vector
%       .bookkeeping             L  — length bookkeeping vector
%       .approximation           Approximation coefficients at deepest level
%       .details                 Cell {1×L}: detail coefficients at each level
%
%     Energy per subband
%       .approximationEnergy     sum(A²)
%       .detailEnergy            1×L vector  sum(D²) per level
%       .totalWaveletEnergy      approximation + all details
%       .approximationEnergyRatio
%       .detailEnergyRatio       1×L vector
%
%     Interpretation
%       .dominantLevel           Level index with the highest detail energy
%       .waveletEntropy          Shannon entropy of the energy distribution
%                                (high → energy spread across scales,
%                                 low  → energy concentrated in one scale)
%
%     Reconstruction (from wrcoef)
%       .reconstructedApproximation  Full-length reconstructed A signal
%       .reconstructedDetails        Cell {1×L}: reconstructed D signals
%       .reconstructedSignal         Sum of all reconstructed components
%       .reconstructionRMSE          RMS reconstruction error (≈ 0 ideally)
%       .reconstructionMAE           Mean absolute reconstruction error
%
%     Frequency bands (approximate, dyadic)
%       .approximationBand       [0, Fs/2^(L+1)]  Hz
%       .detailBands             L×2 matrix: [low, high] Hz per level
%
%   Requirements
%   ------------
%   MATLAB Wavelet Toolbox (wavedec, appcoef, detcoef, wrcoef, wmaxlev)
%
%   Example
%   -------
%       % Impulse — energy should concentrate in high-frequency details
%       Fs = 16000;
%       x  = zeros(16000, 1);
%       x(8000) = 1;
%       result = runWavelet(x, Fs, "db4", 5);
%       disp(result.detailEnergy)
%
%   Viva note
%   ---------
%   DWT is related to STFT through the Heisenberg-Gabor principle.
%   STFT uses a fixed window size (fixed time-freq resolution).
%   DWT uses windows that *scale with frequency*: high-freq → short
%   window (good time resolution); low-freq → long window (good freq
%   resolution).  This makes DWT superior for signals whose features
%   span multiple time scales, such as impulses, steps, and chirps.

    %% ================================================================
    %  1. INPUT VALIDATION
    %  ================================================================

    if nargin < 2
        error("SPANDHAN:Wavelet:MissingArgs", ...
            "runWavelet requires at least x and Fs.");
    end

    if nargin < 3 || isempty(waveletName)
        waveletName = "db4";
    end

    if nargin < 4 || isempty(decompositionLevel)
        decompositionLevel = 5;
    end

    % Force column vector of doubles
    x = double(x(:));

    if isempty(x)
        error("SPANDHAN:Wavelet:EmptySignal", ...
            "Input signal cannot be empty.");
    end

    if any(~isfinite(x))
        error("SPANDHAN:Wavelet:InvalidSignal", ...
            "Input signal must contain finite numeric values.");
    end

    validateattributes(Fs, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, mfilename, 'Fs');

    if ~isscalar(decompositionLevel) || ...
            decompositionLevel < 1   || ...
            decompositionLevel ~= floor(decompositionLevel)
        error("SPANDHAN:Wavelet:InvalidLevel", ...
            "decompositionLevel must be a positive integer.");
    end

    waveletName = string(waveletName);

    %% ================================================================
    %  2. DC REMOVAL
    %  ================================================================
    %
    %  Removing the mean prevents the DC component from concentrating
    %  all energy in the approximation and masking detail structure.

    xOriginal = x;
    xAnalysis = x - mean(x);

    %% ================================================================
    %  3. MAXIMUM DECOMPOSITION LEVEL
    %  ================================================================
    %
    %  wmaxlev() returns the maximum number of levels possible for a
    %  given signal length and wavelet filter length.  Going beyond
    %  this produces coefficients shorter than the filter — meaningless.

    try
        maxLevel = wmaxlev(length(xAnalysis), char(waveletName));
    catch ME
        error("SPANDHAN:Wavelet:ToolboxMissing", ...
            "Cannot determine maximum wavelet level. " + ...
            "Ensure the Wavelet Toolbox is installed.\n%s", ME.message);
    end

    if maxLevel < 1
        error("SPANDHAN:Wavelet:SignalTooShort", ...
            "Signal is too short for wavelet decomposition with '%s'.", ...
            waveletName);
    end

    if decompositionLevel > maxLevel
        warning("SPANDHAN:Wavelet:LevelClamped", ...
            "Requested level %d exceeds maximum %d for this signal. " + ...
            "Clamping to %d.", ...
            decompositionLevel, maxLevel, maxLevel);
        decompositionLevel = maxLevel;
    end

    %% ================================================================
    %  4. DISCRETE WAVELET DECOMPOSITION
    %  ================================================================
    %
    %  wavedec returns:
    %    C  — concatenated coefficient vector [A_L | D_L | … | D_1]
    %    L  — lengths: [len(A_L), len(D_L), …, len(D_1), len(x)]

    [C, L] = wavedec(xAnalysis, decompositionLevel, char(waveletName));

    %% ================================================================
    %  5. EXTRACT APPROXIMATION COEFFICIENTS
    %  ================================================================

    approximation = appcoef(C, L, char(waveletName), decompositionLevel);
    approximation = approximation(:);

    %% ================================================================
    %  6. EXTRACT DETAIL COEFFICIENTS (all levels)
    %  ================================================================

    details = cell(1, decompositionLevel);

    for k = 1:decompositionLevel
        details{k} = detcoef(C, L, k);
        details{k} = details{k}(:);
    end

    %% ================================================================
    %  7. ENERGY PER SUBBAND
    %  ================================================================

    detailEnergy = zeros(1, decompositionLevel);

    for k = 1:decompositionLevel
        detailEnergy(k) = sum(details{k}.^2);
    end

    approximationEnergy = sum(approximation.^2);
    totalWaveletEnergy  = approximationEnergy + sum(detailEnergy);

    %% ================================================================
    %  8. ENERGY RATIOS
    %  ================================================================

    if totalWaveletEnergy > eps
        approximationEnergyRatio = approximationEnergy / totalWaveletEnergy;
        detailEnergyRatio        = detailEnergy        / totalWaveletEnergy;
    else
        approximationEnergyRatio = 0;
        detailEnergyRatio        = zeros(1, decompositionLevel);
    end

    %% ================================================================
    %  9. DOMINANT DETAIL LEVEL
    %  ================================================================
    %
    %  The level with the highest detail energy indicates where the
    %  signal's dominant structural activity occurs.
    %  D1 → high-frequency / transient dominated
    %  D5 → low-frequency / slow variation dominated

    if any(detailEnergy > 0)
        [~, dominantLevel] = max(detailEnergy);
    else
        dominantLevel = NaN;
    end

    %% ================================================================
    %  10. WAVELET ENTROPY (Shannon, energy-based)
    %  ================================================================
    %
    %  Measures how spread the energy is across decomposition levels.
    %  Low entropy  → energy concentrated (tonal / impulse-like)
    %  High entropy → energy distributed across scales (broadband / noise)

    energyDistribution = [approximationEnergy, detailEnergy];
    energySum          = sum(energyDistribution);

    if energySum > eps
        p               = energyDistribution / energySum;
        p               = p(p > 0);            % avoid log2(0)
        waveletEntropy  = -sum(p .* log2(p));
    else
        waveletEntropy = 0;
    end

    %% ================================================================
    %  11. RECONSTRUCT APPROXIMATION AT FULL LENGTH
    %  ================================================================

    reconstructedApproximation = ...
        wrcoef("a", C, L, char(waveletName), decompositionLevel);
    reconstructedApproximation = reconstructedApproximation(:);

    %% ================================================================
    %  12. RECONSTRUCT EACH DETAIL AT FULL LENGTH
    %  ================================================================

    reconstructedDetails = cell(1, decompositionLevel);

    for k = 1:decompositionLevel
        reconstructedDetails{k} = wrcoef("d", C, L, char(waveletName), k);
        reconstructedDetails{k} = reconstructedDetails{k}(:);
    end

    %% ================================================================
    %  13. RECONSTRUCTION VERIFICATION
    %  ================================================================
    %
    %  Summing all reconstructed components must equal xAnalysis.
    %  Any deviation from zero indicates numerical or toolbox issues.

    reconstructedSignal = reconstructedApproximation;

    for k = 1:decompositionLevel
        reconstructedSignal = reconstructedSignal + reconstructedDetails{k};
    end

    reconstructionError = xAnalysis - reconstructedSignal;
    reconstructionRMSE  = sqrt(mean(reconstructionError.^2));
    reconstructionMAE   = mean(abs(reconstructionError));

    %% ================================================================
    %  14. TIME AXIS
    %  ================================================================

    N    = length(xOriginal);
    time = (0 : N - 1)' / Fs;

    %% ================================================================
    %  15. APPROXIMATE FREQUENCY BANDS (dyadic)
    %  ================================================================
    %
    %  Approximation at level L covers: 0  →  Fs / 2^(L+1)
    %  Detail at level k covers:        Fs / 2^(k+1)  →  Fs / 2^k
    %
    %  These are nominal half-band boundaries, not exact filter cutoffs.

    approximationBand = [0, Fs / 2^(decompositionLevel + 1)];

    detailBands = zeros(decompositionLevel, 2);

    for k = 1:decompositionLevel
        detailBands(k, :) = [Fs / 2^(k+1),  Fs / 2^k];
    end

    %% ================================================================
    %  16. ASSEMBLE RESULT STRUCTURE
    %  ================================================================

    result = struct();

    % --- Time-domain data ---
    result.input          = xOriginal;
    result.analysisSignal = xAnalysis;
    result.time           = time;

    % --- Configuration ---
    result.samplingFrequency  = Fs;
    result.waveletName        = waveletName;
    result.decompositionLevel = decompositionLevel;
    result.maximumLevel       = maxLevel;

    % --- Raw DWT outputs ---
    result.coefficients = C;
    result.bookkeeping  = L;
    result.approximation = approximation;
    result.details       = details;

    % --- Energy ---
    result.approximationEnergy      = approximationEnergy;
    result.detailEnergy             = detailEnergy;
    result.totalWaveletEnergy       = totalWaveletEnergy;
    result.approximationEnergyRatio = approximationEnergyRatio;
    result.detailEnergyRatio        = detailEnergyRatio;

    % --- Interpretation ---
    result.dominantLevel   = dominantLevel;
    result.waveletEntropy  = waveletEntropy;

    % --- Reconstructions ---
    result.reconstructedApproximation = reconstructedApproximation;
    result.reconstructedDetails       = reconstructedDetails;
    result.reconstructedSignal        = reconstructedSignal;
    result.reconstructionRMSE         = reconstructionRMSE;
    result.reconstructionMAE          = reconstructionMAE;

    % --- Frequency bands ---
    result.approximationBand = approximationBand;
    result.detailBands       = detailBands;

end
